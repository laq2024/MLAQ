import os
import json
import copy
import time
import torch
import shutil
import openai
import tiktoken
import os.path as path
from typing import Dict
from openai import OpenAI

from prompting.lang_tree import LangTree, preprocess_state, preprocess_action
from rocobench.envs import EnvState


class ModelPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent
    """
    def __init__(
            self,
            env, parser, feedback_manager,
            max_tokens: int = 512,
            debug_mode: bool = False,
            use_waypoints: bool = False,
            robot_name_map: Dict[str, str] = {"panda": "Bob"},
            num_replans: int = 3,
            max_calls_per_round: int = 10,
            use_history: bool = True,
            use_feedback: bool = True,
            temperature: float = 0,
            llm_source: str = "gpt-4",
            load: bool = False,
            skip_check_memory: bool = False
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = []
        self.latest_chat_history = []
        self.max_calls_per_round = max_calls_per_round
        self.temperature = temperature
        self.llm_source = llm_source
        self.memory_path = 'data/{}/memory.pth'.format(self.env.name)
        self.skip_check_memory = skip_check_memory
        self.original_state_desp = None

        self.client = OpenAI()
        self.token_encoder = tiktoken.encoding_for_model(llm_source)
        self.last_end_step = 100

        # add the assumed tokens of each round into this list
        self.assumed_token = {
            'input_token': [],
            'output_token': []
        }
        # this memory ratio is used to statistics the calling frequency of the memory's transition data
        self.memory_ratio = {
            'llm_count': [],
            'memory_count': []
        }
        # this dict is used for temporary storage of the assumed tokens of each round
        self.temp_assumed_token = {
            'input_token': 0,
            'output_token': 0
        }
        self.temp_memory_ratio = {
            'llm_count': 0,
            'memory_count': 0
        }

        if load:
            try:
                self.memory = torch.load(self.memory_path)
            except:
                self.memory = {
                    'prediction': {},
                    'policy': {},
                }
        else:
            self.memory = {
                'prediction': {},
                'policy': {},
            }

        self.temporary_memory = {
            'prediction': {},
            'policy': {},
        }
        self.state = None
        self.action = None
        self.lang_tree = LangTree(load=load, if_temp_tree=False, tree_path='data/{}/lang_tree.pth'.format(self.env.name))
        self.temp_lang_tree = LangTree(load=False, if_temp_tree=True)
        self.case_dir, self.simulate_save_dir, self.interact_save_dir = None, None, None
        self.round_save_dir, self.env_round_save_dir = None, None
        self.round_index = 1
        self.env_round_index = 1
        self.interaction_failure = 0
        self.imagination_failure = 0

        assert llm_source in ["gpt-4", "gpt-3.5-turbo", 'gpt-4-1106-preview', 'gpt-4-0125-preview',  "claude"], \
            f"llm_source must be one of [gpt4, gpt-3.5-turbo, gpt-4-1106-preview, gpt-4-0125-preview, claude], got {llm_source}"

    def initial_temp_lang_tree(self, initial_obs, target_state_desp):
        # initial_obs is the initial observation of the current testing case or the last failed observation
        # if the reset_temp_tree is True, it means that the temp_lang_tree should be reset (at the beginning of the case experiment)
        # if the reset_temp_tree is False, it means that the temp_lang_tree should not be reset (when the imagination fails)
        state_desp = self.env.get_state_prompt(initial_obs)
        self.temp_lang_tree = LangTree(load=False, if_temp_tree=True)

        self.temp_lang_tree.build_virtual_node(state_desp)
        self.temp_lang_tree.get_node(state_desp).visit_num += 1

        path_exist = self.check_path_exist(state_desp, target_state_desp=target_state_desp)
        if path_exist:
            self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)
            reach_target = False
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))
                action, next_state_node = state_node.get_best()
                self.temp_lang_tree.update(transition=[state_node.state, action, next_state_node.state])
                self.temp_lang_tree.get_node(next_state_node.state).visit_num += 1

                self.temp_lang_tree.build_virtual_node(next_state_node.state)

                state_desp = next_state_node.state

                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

                    self.temp_lang_tree.update(transition=[state_desp, 'nothing', 'Success'])
                    success_node = self.temp_lang_tree.get_node('Success')
                    self.temp_lang_tree.get_node('Success').visit_num += 1

                    self.temp_lang_tree.remove_node(state_desp, 'action_virtual', virtual=True)
                    success_node.value = 1

            self.temp_lang_tree.compute_ucb_values(loop=20)

    def fill_temp_lang_tree(self, original_state_desp, target_state_desp):
        self.temp_lang_tree.build_virtual_node(original_state_desp)
        path_exist = self.check_path_exist(original_state_desp, target_state_desp=target_state_desp)

        if path_exist:
            self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)
            reach_target = False
            state_desp = original_state_desp
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))
                action, next_state_node = state_node.get_best()
                self.temp_lang_tree.update(transition=[state_node.state, action, next_state_node.state])
                self.temp_memory_ratio['memory_count'] += 1
                if self.temp_lang_tree.get_node(state_node.state).visit_num == 0:
                    self.temp_lang_tree.get_node(state_node.state).visit_num += 1
                if self.temp_lang_tree.get_node(next_state_node.state).visit_num == 0:
                    self.temp_lang_tree.get_node(next_state_node.state).visit_num += 1

                self.temp_lang_tree.build_virtual_node(next_state_node.state)

                state_desp = next_state_node.state

                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

                    self.temp_lang_tree.update(transition=[state_desp, 'nothing', 'Success'])
                    success_node = self.temp_lang_tree.get_node('Success')
                    self.temp_lang_tree.remove_node(state_desp, 'action_virtual', virtual=True)
                    success_node.value = 1
                    if success_node.visit_num == 0:
                        success_node.visit_num += 1

        self.temp_lang_tree.compute_ucb_values(loop=20)

    def get_end_step(self, target_state_desp):
        self.temp_lang_tree.build_virtual_node(self.original_state_desp)
        path_exist = self.check_path_exist(self.original_state_desp, target_state_desp=target_state_desp)
        end_step = 0
        if path_exist:
            self.lang_tree.compute_values(target_state=preprocess_state(target_state_desp), loop=20)
            reach_target = False
            state_desp = self.original_state_desp
            while not reach_target:
                state_node = self.lang_tree.get_node(preprocess_state(state_desp))
                action, next_state_node = state_node.get_best()
                end_step += 1
                state_desp = next_state_node.state

                if preprocess_state(state_desp) == preprocess_state(target_state_desp):
                    reach_target = True

        return end_step

    def round_reset(self):
        self.state = None
        self.action = None

        self.temporary_memory = copy.deepcopy(self.memory)

    def case_reset(self):
        self.interaction_failure = 0
        self.imagination_failure = 0
        self.last_end_step = 100

        self.temporary_memory = copy.deepcopy(self.memory)
        self.assumed_token = {
            'input_token': [],
            'output_token': []
        }
        self.memory_ratio = {
            'llm_count': [],
            'memory_count': []
        }
        self.round_index = 1
        self.original_state_desp = None

    def get_summary(self):
        # return the assumed token and memory ratio until the end of this round
        return self.assumed_token, self.memory_ratio

    def _summarize_this_round(self, original_state_desp, target_state_desp, interrupt_flag=False):
        # summarize the token and memory ratio of this round
        # return if the test should be conducted after this round

        if len(self.temp_lang_tree.nodes) > 40:
            return True

        if self.imagination_failure >= 3:
            return True

        path_exist = self.check_path_exist(original_state_desp, target_state_desp)
        current_end_step = self.get_end_step(target_state_desp)
        if path_exist:
            optimized_flag = False
            if current_end_step < self.last_end_step:
                optimized_flag = True

            self.last_end_step = current_end_step

        if interrupt_flag and path_exist:
            return True

        if self.temp_assumed_token['input_token'] == 0:
            # if the assumed token is 0, it means that the LLM has not been called in this round
            # therefore, we should not add the assumed token of this round into the list
            if self.round_index == 1:
                if len(self.assumed_token['input_token']) == 0:
                    self.assumed_token['input_token'].append(0)
                    self.assumed_token['output_token'].append(0)
                    self.memory_ratio['llm_count'].append(0)
                    self.memory_ratio['memory_count'].append(0)
                    return True
                else:
                    self.round_index += 1
                    return False
            else:
                shutil.rmtree(self.round_save_dir)
                return False
        else:
            # if the assumed token is not 0, it means that the LLM has been called in this round
            self.assumed_token['input_token'].append(self.temp_assumed_token['input_token'])
            self.assumed_token['output_token'].append(self.temp_assumed_token['output_token'])
            self.memory_ratio['llm_count'].append(self.temp_memory_ratio['llm_count'])
            self.memory_ratio['memory_count'].append(self.temp_memory_ratio['memory_count'])
            self.round_index += 1
            if path_exist:
                # if the current end step is the optimal steps, we should conduct the test
                if current_end_step <= self.env.optimal_steps:
                    return True

                if optimized_flag:
                    return True
                else:
                    if self.round_index >= 8:
                        return True
                    else:
                        return False
            else:
                return False

    def set_case_dir(self, save_dir):
        assert save_dir is not None, "save_dir cannot be None"
        self.case_dir = save_dir
        if not os.path.exists(self.case_dir):
            os.makedirs(self.case_dir)

        self.simulate_save_dir = path.join(self.case_dir, "simulate")
        if not os.path.exists(self.simulate_save_dir):
            os.makedirs(self.simulate_save_dir)

        self.interact_save_dir = path.join(self.case_dir, "interact")
        if not os.path.exists(self.interact_save_dir):
            os.makedirs(self.interact_save_dir)

    def set_round_dir(self, round_save_dir):
        assert round_save_dir is not None, "round_save_dir cannot be None"
        self.round_save_dir = round_save_dir
        if not os.path.exists(self.round_save_dir):
            os.makedirs(self.round_save_dir)

    def set_env_round_dir(self, env_round_save_dir):
        assert env_round_save_dir is not None, "env_round_save_dir cannot be None"
        self.env_round_save_dir = env_round_save_dir
        if not os.path.exists(self.env_round_save_dir):
            os.makedirs(self.env_round_save_dir)

    def log(self, message, file_name="/log.txt", if_test=False):
        if if_test:
            save_file_name = self.env_round_save_dir + file_name
        else:
            save_file_name = self.round_save_dir + file_name

        with open(save_file_name, "a") as f:
            f.write(message + "\n")

        print(message)

    def _extract_action(self, action_prompt: str, info=None) -> str:
        prompt = f"""There is a description of an action space description and the action space definition as follows:
[----- Action Space Description Begin -----]
{self.env.get_action_space_prompt()}.
[----- Action Space Description End -----]
Now, I will provide you the decision process of the agent as follows:
[----- Decision Process Begin -----]
{action_prompt}
[----- Decision Process End -----]
Now, read the above paragraphs especially the conclusion, and help me to extract the [Action] in the format of [Action Space Definition]. If there is no action, please output [ERROR].
The only thing you should output is the [Action] and do not include any other information. Please note that do not include any marks like quotations."""
        response = self.new_query_once(prompt, use_stream=True, info=info)

        return response

    def _extract_state(self, state_prompt: str, info=None) -> str:
        prompt = f"""There is a description of a state space as follows:
[----- State Space Description Begin -----]
{self.env.get_state_space_prompt()}.
[----- State Space Description End -----]
Now, I will provide you the predicting process of the agent as follows:
[----- Predicting Process Begin -----]
{state_prompt}
[----- Predicting Process End -----]
Now, read the above paragraphs especially the conclusion, and help me to extract the [State] in the format of [State Space Definition].
The only thing you should output is the [State] and do not include any other information. Please note that do not include any marks like quotations."""
        response = self.new_query_once(prompt, use_stream=True, info=info)

        return response

    def _get_predictor_prompt(self, state, action, info) -> dict:
        action_space_desp = self.env.get_action_space_prompt()
        state_space_desp = self.env.get_state_space_prompt()
        task_prompt = self.env.get_task_prompt()
        memory_prompt = '[Prediction Mistakes]\n'
        if preprocess_state(state) in self.memory['prediction'].keys():
            for memory in self.memory['prediction'][preprocess_state(state)]:
                wrong_prediction, true_prediction = memory
                memory_prompt += f"""In the current state and action, you have outputted a wrong prediction. The prediction is as follows:
{wrong_prediction}
The true (correct) prediction is:
{true_prediction}
"""
        else:
            memory_prompt += 'No prediction mistakes.'

        system_prompt = f"""You are a state predictor. You will be given a [State] and an [Action] as input. should predict the [State] after the [Action] is applied to the given [State]. You should follow the [Predicting Instruction] to predict the [State].
At first, you should first repeat the state and action as follows: [Repeat current State] and [Repeat action].
  + [Repeat current State]: {self.env.get_repeat_template()['state']}.
  + [Repeat action]: {self.env.get_repeat_template()['action']}.
{self.env.get_predicting_prompt()}
"""
        system_prompt = f"""{system_prompt}\nThe following lists the [Task Description], [State Space Definition], and [Action Space Definition].
{task_prompt}
{state_space_desp}
{action_space_desp}
{memory_prompt}
{info['feedbacks']}
Now, output your prediction below. You should follow the [Predicting Instruction] strictly and think step-by-step."""

        user_prompt = f"""Now, you should predict the next [State] after the following [Action] is applied to the given [State]. You have to follow the instructions strictly rather than your own thoughts.
{state}
{action}
"""
        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def _get_action_checker_prompt(self, state: str, action: str) -> dict:
        action_space_desp = self.env.get_action_space_prompt()
        state_space_desp = self.env.get_state_space_prompt()
        task_prompt = self.env.get_task_prompt()
        memory_prompt = '[Action Mistakes]\n'
        if preprocess_state(state) in self.memory['policy'].keys():
            for memory in self.memory['policy'][preprocess_state(state)]:
                wrong_action, feedback = memory
                memory_prompt += f"""In the current state, you mistakenly believed that a wrong action was the correct one. The action is as follows:
{wrong_action}
The environment provides the [Feedback] as follows:
{feedback}
"""
        else:
            memory_prompt += 'No action mistakes.'

        system_prompt = f"""You are an action checker. You should check if the [Action] is valid based on the given [State]. You should follow the [Checker Instruction] to check the [Action]. 
At first, you should first repeat the state and action as follows: [Repeat current State], [Repeat target State], and [Repeat action].
  + [Repeat current State]: {self.env.get_repeat_template()['state']}.
  + [Repeat action]: {self.env.get_repeat_template()['action']}.
[Checker Instruction]
1. You should follow the [Task Description] carefully to check the [Action].
2. You should output your thinking process step-by-step to follow these steps: [Action Constraints Check], [Conclusion Justification], and [Checker Conclusion].
  + [Action Constraints Check]: follow the check instructions in [Detailed Constraints for Action] to check the [Action] step-by-step. Take the mistakes in the [Action Mistakes] into consideration.
  + [Temporary Conclusion]: output your temporary conclusion about the validity of the [Action]. You should provide a reason for your conclusion. 
  + [Conclusion Justification]: check if the checking process above is complete and check if the temporary conclusion is correct.
  + [Checker Conclusion]: output your final conclusion about the validity of the [Action]. You should provide a reason for your conclusion. If the [Action] is valid, you should output [VALID]. I will use this to extract your results in the following steps.
"""
        system_prompt = f"{system_prompt}\nThe following lists the [Task Description], [State Space Definition], and [Action Space Definition]. \n{task_prompt}\n{state_space_desp}\n{action_space_desp}\n{memory_prompt}\n"
        system_prompt += f"""Now, output your check conclusion below. You should follow the [Checker Instruction] strictly and think step-by-step. """

        user_prompt = f"""Now you should check the following [Action] based on the given [State]. You have to check the instructions step-by-step strictly rather than your own thoughts.
{state}
{action}
"""
        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def _get_prediction_checker_prompt(self, state: str, action: str, prediction: str) -> dict:
        action_space_desp = self.env.get_action_space_prompt()
        state_space_desp = self.env.get_state_space_prompt()
        task_prompt = self.env.get_task_prompt()
        memory_prompt = '[Prediction Mistakes]\n'
        search_feature = preprocess_state(state) + preprocess_action(action)
        if search_feature in self.memory['prediction'].keys():
            for memory in self.memory['prediction'][search_feature]:
                wrong_prediction, true_prediction = memory
                memory_prompt += f"""In the current state and action, you mistakenly believed that a wrong prediction was the correct one. The prediction is as follows:
{wrong_prediction}
The true (correct) prediction is:
{true_prediction}
"""
        else:
            memory_prompt += 'No prediction mistakes.'

        system_prompt = f"""You are a prediction checker. You will receive a [State], an [Action], and the subsequent [Prediction]. You should check if the [Prediction] is valid based on the given [State] and [Action]. Follow the [Checker Instruction] strictly to check the [Prediction].
[Checker Instruction]
At first, you should first repeat the state and action as follows: [Repeat state] and [Repeat action].
  + [Repeat State]: {self.env.get_repeat_template()['state']}.
  + [Repeat action]: {self.env.get_repeat_template()['action']}.
The action has been checked by the action checker. You should not check the [Action] again. You should output your thinking process step-by-step by following theses steps: 
  + [Prediction Format Check]: check if the [Prediction] state meets the [State Template] and satisfies the [Detailed Constraints for State]. Follow the <check items> in the [Detailed Constraints for State] to check the [Prediction] step by step.
  + [Prediction Rule Check]: follow the [Predicting Instruction] to check if the [Prediction] state satisfies the predicting rules. Take the mistakes in the [Prediction Mistakes] into consideration.
  + [Temporary Conclusion]: output your temporary conclusion about the validity of the [Prediction] in the end of this part. 
  + [Conclusion Justification]: check again if the [Prediction] satisfies the predicting rules.
  + [Final Conclusion]: output your final conclusion about the validity of the [Prediction]. You should provide a reason for your conclusion.
    - In the [Checker Conclusion], if the [Prediction] is valid, you should output [CORRECT]. I will use this to extract your results. If the [Prediction] is invalid, you should output your reason for the invalidity.
"""
        system_prompt = f"{system_prompt}\nThe following lists the [Task Description], [State Space Definition], [Action Space Definition], and [Predicting Instruction]. \n{task_prompt}\n{state_space_desp}\n{action_space_desp}\n{self.env.get_predicting_prompt()}\n{memory_prompt}\n"
        system_prompt += f"""Now, output your check conclusion. You should follow the [Checker Instruction] strictly and think step-by-step"""

        user_prompt = f"""Now, check the following [Prediction]. You have to check the instructions one-by-one and step-by-step strictly according to [Checker Instruction] rather than your own thoughts. 
The original state is:
{state}
The action is:
{action}
The prediction is:
{prediction}"""

        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def _get_policy_prompt(self, state, next_state, info) -> dict:
        next_state = next_state.replace('[State]', '[Target State]')
        action_space_desp = self.env.get_action_space_prompt()
        state_space_desp = self.env.get_state_space_prompt()
        task_prompt = self.env.get_task_prompt()
        memory_prompt = '[Policy Mistakes]\n'
        if preprocess_state(state) in self.memory['policy'].keys():
            for memory in self.memory['policy'][preprocess_state(state)]:
                wrong_action, feedback = memory
                memory_prompt += f"""In the current state, you have chosen a wrong [Action]:
{wrong_action}
The environment provides the [Feedback] as follows:
{feedback}
"""
        else:
            memory_prompt += 'No policy mistakes.'

        system_prompt = f"""You are a centralized multi-agent planner. You have received a [State] from the external environment. Now, you need to output an action to reach the target [State] from current [State]. Follow the [Detailed Instruction for Policy] strictly.
[Detailed Instruction for Policy]
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content. 
1. You should think step by step, and make sure the [Action] does not break the [Detailed Constraints for Action].
2. Follow the [Task Description] Carefully. You should follow the task description to plan the [Action].
3. You should follow the [Recipe Order] to plan the [Action]. The format of action is described in [Detailed Constraints for Action].
4. You should output your thinking process step-by-step in the following order. You have to follow these steps one by one to plan the [Action]: [Action Planning], [Action Conclusion], [Action Constraints Check], [Forbidden Action Check], and [Action Output].
+ [1. Action Planning]: First output [Action Planning], then follow the [Action Planning Instructions] to plan the [Action] of the multi-agent system step by step and list the thinking process.
+ [2. Action Conclusion]: Conclude the [Action] in the format of [Detailed Constraints for Action].
+ [3. Action Constraints Check]: Follow the steps in [Detailed Constraints for Action] to check the [Action] step by step.
+ [4. Action Revise]: If there is no incorrect checking result in the [Action Constraints Check], you can pass to the [Forbidden Action Check]. Otherwise, you should revise the action. Take the feedback from [Action Constraints Check] into consideration, and follow these steps to revise the action: [New Action Planning] and [New Action Constraints Check].
  - Run a loop to execute the [Action Planning], [Action Constraints Check], and [Action Revise] until the [Action] is correct.
+ [5. Forbidden & Mistake Action Check]: I will provide some forbidden joint actions, and the robots cannot choose the forbidden and mistake actions. Note that what is forbidden are the joint actions of the multi-agent system, rather than the single agent actions that appear in these joint actions.
  - list the chosen action.
  - repeat all actions and feedbacks in the [Forbidden Actions], [Policy Mistakes], and [Temporary Mistakes]: [[action, feedback], ...]
  - check if the [Action] is in the [Temporary Mistakes] one by one. If so, you should first judge whether the [Action] is correct or not. If the [Action] is correct, you can pass to the [Action Output] part. Otherwise, you should revise the [Action] step by step.
  - check if the [Action] is in the [Forbidden Actions] or [Policy Mistakes] one by one.
  If the [Action] is not in the [Forbidden Actions] and [Policy Mistakes]:
    - Pass to the [Action Output] part.
  Else:
    - Follow these instructions to plan a new [Actions] step by step and list the thinking process:
      - Go back to a new [Action Planning] and [Action Constraints Check] process again. You should note that what is forbidden are the joint actions rather than the single agent actions that appear in these joint actions. Therefore, the individual actions of each agent in the [Forbidden Actions] can still be chosen.
      - [Forbidden Action Check]: check if the [Action] satisfies the [Detailed Constraints for Action].
      - Re-plan the [Action] until a new [Action] is obtained or there is no available action anymore. You can try 2-nd, 3-rd, 4-th ... Plan to get the final [Action].
+ [6. Action Output]: output the final action in the format of [Detailed Constraints for Action]. If there is no available action anymore, output [ERROR] to denote that there is no available action anymore.

You can take the [Plan Example] as a reference. This example only contains the outputted action.
{self.env.get_examples()['plan']}
"""
        system_prompt = f"{system_prompt}\nThe following lists the [Task Description], [State Space Definition], and [Action Space Definition]. \n{task_prompt}\n{state_space_desp}\n{action_space_desp}\n{memory_prompt}\n"
        if len(info['feedbacks']) > 0:
            system_prompt += f"""
[Temporary Mistakes]
You have made {len(info['feedbacks'])} mistakes. The following actions are the wrong actions you have made.
These feedbacks are from the previous imagination, and may not be correct. However, you still need to pay attention to them and avoid making the same or similar mistakes."""
            for i, feedback in enumerate(info['feedbacks']):
                system_prompt += f"""
The wrong action {i + 1} is as follows:
{feedback['action']}
The feedback is as follows:
{feedback['feedback']}"""
        system_prompt += f"""Now, output your action below. You should follow the [Detailed Instruction for Policy] strictly and think step-by-step to plan the action."""
        system_prompt += f"""\n{self.get_forbidden_action_prompt(info['forbidden_actions'])}"""

        user_prompt = f"""Now you should output your action according to the following current state and target state. 
[Plan]
{state}
{next_state}"""

        return {'system_prompt': system_prompt, 'user_prompt': user_prompt}

    def get_role_prompt(self, role: str, state=None, action=None, next_state=None, info=None) -> dict:
        # role: predictor, action_checker, prediction_checker, policy
        if role == 'predictor':
            prompt = self._get_predictor_prompt(state, action, info)
        elif role == 'action_checker':
            prompt = self._get_action_checker_prompt(state, action)
        elif role == 'prediction_checker':
            prompt = self._get_prediction_checker_prompt(state, action, next_state)
        elif role == 'policy':
            prompt = self._get_policy_prompt(state, next_state, info)
        else:
            raise ValueError(
                f"role must be one of ['predictor', 'action_checker', 'prediction_checker', 'policy'], got {role}"
            )

        return prompt

    def estimate_state_prompt(self, obs: EnvState) -> str:
        return ''

    def update_memory_policy(self, state: str, action: str, feedback: str, temporary: bool = False):
        # update the memory of the policy
        if not temporary:
            if preprocess_state(state) not in self.memory['policy'].keys():
                self.memory['policy'][preprocess_state(state)] = [[action, feedback]]
                self.temporary_memory['policy'][preprocess_state(state)] = [[action, feedback]]
            else:
                self.memory['policy'][preprocess_state(state)].append([action, feedback])
                self.temporary_memory['policy'][preprocess_state(state)].append([action, feedback])

            torch.save(self.memory, self.memory_path)
        else:
            if preprocess_state(state) not in self.temporary_memory['policy'].keys():
                self.temporary_memory['policy'][preprocess_state(state)] = [[action, feedback]]
            else:
                self.temporary_memory['policy'][preprocess_state(state)].append([action, feedback])

    def update_memory_prediction(self, transition, true_transition, temporary: bool = False):
        if not temporary:
            state, action, next_state = transition
            true_next_state = true_transition[-1]
            search_feature = preprocess_state(state) + preprocess_action(action)
            if search_feature not in self.memory['prediction'].keys():
                self.memory['prediction'][search_feature] = [[next_state, true_next_state]]
                self.temporary_memory['prediction'][search_feature] = [[next_state, true_next_state]]
            else:
                self.memory['prediction'][search_feature].append([next_state, true_next_state])
                self.temporary_memory['prediction'][search_feature].append([next_state, true_next_state])

            torch.save(self.memory, self.memory_path)
        else:
            state, action, next_state = transition
            true_next_state = true_transition[-1]
            search_feature = preprocess_state(state) + preprocess_action(action)
            if search_feature not in self.temporary_memory['prediction'].keys():
                self.temporary_memory['prediction'][search_feature] = [[next_state, true_next_state]]
            else:
                self.temporary_memory['prediction'][search_feature].append([next_state, true_next_state])

    def get_forbidden_action_prompt(self, forbidden_actions: list):
        prompt = '[Forbidden Actions]\n'
        if len(forbidden_actions) == 0:
            prompt += 'No forbidden actions.'
        else:
            prompt += 'The following lists the forbidden actions:\n'
            for i, action in enumerate(forbidden_actions):
                prompt += '[Action {}]\n{}'.format(i + 1, action)

        prompt += '\nYou should try your best to output an action that is not in the list above. '

        return prompt

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret

    def get_imagine_flag(self, state_desp):
        result = False
        # if the current state is not in the temporary language tree
        if self.temp_lang_tree.get_node(state_desp) is None:
            result = True
        else:
            # if the current state has no children (only the virtual action)
            if self.temp_lang_tree.get_node(state_desp).num_children == 1:
                result = True

        return result

    def check_path_exist(self, state_desp, target_state_desp):
        lang_tree = copy.deepcopy(self.lang_tree)
        state_node = lang_tree.get_node(state_desp)
        target_state_node = lang_tree.get_node(target_state_desp)
        if target_state_node is None or state_node is None:
            path_exist = False
        else:
            lang_tree.compute_values(target_state=target_state_desp, loop=10)
            if state_node.value > 0.5:
                path_exist = True
            else:
                path_exist = False

        return path_exist

    def decision_with_model(self, obs: EnvState, path_exist: bool = False):
        ready_to_execute, failed = False, False
        state_desp = self.env.get_state_prompt(obs)
        if self.state is not None:
            check_pass, transitions = self.lang_tree.check([self.state, self.action, state_desp])
            if not check_pass:
                self.update_memory_prediction(transitions['predicted_transition'], transitions['true_transition'])

        trial = 0
        while not ready_to_execute:
            # imagine_flag: denote whether current state is in language tree or whether current state has children
            imagine_flag = self.get_imagine_flag(state_desp)
            if failed or imagine_flag:
                path_exist = self.check_path_exist(state_desp, target_state_desp=self.env.target_state_desp)
                # assert path_exist is False, "The path should not exist after the imagination fails or the current state is not in the language tree."
                self.log("-------------- Previous Imagination faces a failure. Start a temporary imagination to make the decision  --------------", if_test=True)
                self.interaction_failure += 1

                # if the imagination fails more than 3 times or the temporary language tree is too large but still have no path to the target state
                # then, the decision process should be terminated
                if (not path_exist and self.interaction_failure > 3) or (not path_exist and len(self.temp_lang_tree.nodes) > 40):
                    self.log("-------------- The temporary imagination fails more than 3 times. The decision process is terminated. --------------", if_test=True)
                    break

                self.imagine_with_ucb(obs=obs, horizon=12)
                path_exist = self.check_path_exist(state_desp, target_state_desp=self.env.target_state_desp)

            if path_exist:
                # there is a path from current state to target state (from pre-imagine or temp-imagine)
                lang_tree = copy.deepcopy(self.lang_tree)
                lang_tree.compute_values(target_state=self.env.target_state_desp, loop=100)
                state_node = lang_tree.get_node(state_desp)
                action, _ = state_node.get_best()
            else:
                # there is no path from current state to target state (from pre-imagine)
                # temp-imagine fails to reach target state
                if self.get_imagine_flag(state_desp):
                    # the current state is not in the language tree or the current state has no children
                    self.log("After the imagination, the current state still has no available children.", if_test=True)
                    continue
                else:
                    state_node = self.temp_lang_tree.get_node(state_desp)
                    action = state_node.children[-1]['action']

            if self.parser is not None:
                parse_succ, parsed_str, llm_plans = self.parser.parse(obs, action)
            else:
                parse_succ, parsed_str, llm_plans = True, '', action

            action_feedback = None
            error_info = {
                'task_error': False,
                'dynamics_error': False,
                'IK_error': False
            }
            if not parse_succ:
                self.log('Failed to parse action: {}'.format(action), if_test=True)
                action_feedback = f"""
This previous [Action] failed to parse!: '{action}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False
                failed = True
            else:
                ready_to_execute = True
                if self.parser is not None:
                    assert type(llm_plans) == list, "llm_plans must be a list when the parser is not none"
                    for j, llm_plan in enumerate(llm_plans):
                        ready_to_execute, env_feedback, error_info = self.feedback_manager.give_feedback(obs, llm_plan)
                        if not ready_to_execute:
                            self.log('Failed to execute action: {}'.format(action), if_test=True)
                            action_feedback = env_feedback
                            if 'IK' in env_feedback:
                                error_info['IK_error'] = True
                            failed = True
                            break
                else:
                    assert type(llm_plans) == str, "llm_plans must be a string when the parser is none"
                    ready_to_execute, env_feedback, error_info = self.feedback_manager.give_feedback(obs, llm_plans)
                    if not ready_to_execute:
                        self.log('Failed to execute action: {}'.format(action), if_test=True)
                        action_feedback = env_feedback
                        failed = True

            if not ready_to_execute:
                self.fill_temp_lang_tree(original_state_desp=state_desp, target_state_desp=self.env.target_state_desp)
                self.log(action_feedback, if_test=True)
                if error_info['task_error'] and error_info['dynamics_error']:
                    raise ValueError("Both task error and dynamics error occur.")

                if 'IK_error' in error_info.keys():
                    if error_info['IK_error']:
                        return None, None, None

                if error_info['task_error']:
                    # the error is caused by the task specific restrictions
                    self.update_memory_policy(state_desp, action, action_feedback, temporary=True)

                if error_info['dynamics_error']:
                    # the error is caused by the dynamics of the environment, so the language tree should be updated
                    self.log("Remove the current state and its children from the language tree.", if_test=True)
                    self.lang_tree.remove_node(state_desp, action)
                    self.temp_lang_tree.remove_node(state_desp, action)
                    self.update_memory_policy(state_desp, action, action_feedback, temporary=False)

            trial += 1

        self.state = self.env.get_state_prompt(obs)
        self.action = action

        return ready_to_execute, llm_plans, path_exist

    def new_query_once(self, prompt, max_query=10, use_stream=False, info=None):
        if type(prompt) == str:
            messages = [
                {"role": "system", "content": prompt},
            ]
            input_str = prompt
        else:
            messages = [
                {"role": "system", "content": prompt['system_prompt']},
                {"role": "user", "content": prompt['user_prompt']},
            ]
            input_str = prompt['system_prompt'] + prompt['user_prompt']

        input_token = len(self.token_encoder.encode(input_str))

        start_time = time.time()
        if use_stream:
            for _ in range(max_query):
                response = ''
                try:
                    stream = self.client.chat.completions.create(
                        model=self.llm_source,
                        messages=messages,
                        temperature=self.temperature,
                        stream=True)
                    for chunk in stream:
                        if len(chunk.choices) > 0:
                            if chunk.choices[0].delta.content is not None:
                                # print(chunk.choices[0].delta.content, end='')
                                response += chunk.choices[0].delta.content
                    break
                except:
                    print("API error, try again")
                    continue
        else:
            response = None
            for _ in range(max_query):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.llm_source,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    response = response['choices'][0]['message']["content"]
                    break
                except:
                    print("API error, try again")
                continue

        output_token = len(self.token_encoder.encode(response))

        end_time = time.time()
        print("Time elapsed: {:.2f}".format(end_time - start_time))

        if info is not None:
            assert 'role' in info.keys(), "info must have key 'role'"
            assert 'timestep' in info.keys(), "info must have key 'timestep'"
            assert 'id' in info.keys(), "info must have key 'id'"
            assert 'property' in info.keys(), "info must have key 'property'"
            save_dir = path.join(self.round_save_dir, f"timestep_{info['timestep']}")
            save_dir = path.join(save_dir, f"{info['role']}_query")
            save_dir = path.join(save_dir, f"trial_{info['id']}")
            if not path.exists(save_dir):
                os.makedirs(save_dir)

            with open(path.join(save_dir, "{}_prompt.txt".format(info['property'])), "w") as f:
                if type(prompt) == str:
                    f.write(prompt)
                else:
                    f.write(prompt['system_prompt'] + prompt['user_prompt'])

            with open(path.join(save_dir, "{}_response.txt".format(info['property'])), "w") as f:
                f.write(response)

        self.temp_assumed_token['input_token'] += input_token
        self.temp_assumed_token['output_token'] += output_token

        return response

    def imagine_with_ucb(self, obs, horizon: int = 10, interrupt_flag=False):
        original_state_desp = self.env.get_state_prompt(obs)
        target_state = self.env.target_state_desp
        self.fill_temp_lang_tree(original_state_desp=original_state_desp, target_state_desp=target_state)

        # set the stop flag, this flag is used to stop the imagination process and turn to an interaction test
        stop_flag = False
        while not stop_flag:
            # initial save dir
            round_save_dir = path.join(self.simulate_save_dir, f"round_{self.round_index}")
            self.set_round_dir(round_save_dir)

            # initialize the original state node
            original_state_node = self.temp_lang_tree.get_node(original_state_desp)
            original_state_node.visit_num += 1
            final, failed, restart = False, False, False
            t, llm_count = 0, 0

            # initialize the data for the round
            self.temp_assumed_token = {'input_token': 0, 'output_token': 0}
            self.temp_memory_ratio = {'llm_count': 0, 'memory_count': 0}

            self.log(f"-------------- Start {self.round_index}th round imagination --------------")
            self.log('Original state:\n' + original_state_desp)
            state_desp = original_state_desp
            while (final is False) and (llm_count < horizon):
                self.log('---------- Time step {} ----------'.format(t+1))
                # get the best child and its action for the current state node
                state_node = self.temp_lang_tree.get_node(state_desp)
                assert state_node is not None, f"state_node is None, state_desp:{state_desp}"
                action, best_child = state_node.get_best(with_ucb=True)
                if action == 'action_virtual':
                    # if the best action is virtual, then the algorithm should use the llm to get the next state
                    self.log('----- Use LLM -----')
                    llm_count += 1
                    temp_forbidden_actions = [child['action'] for child in state_node.children]
                    try:
                        temp_forbidden_actions.remove('action_virtual')
                        temp_forbidden_actions.remove(preprocess_action('ERROR'))
                    except:
                        pass

                    if len(temp_forbidden_actions) > 0:
                        forbidden_actions = copy.deepcopy(temp_forbidden_actions)
                    else:
                        forbidden_actions = []

                    # if the virtual node is chosen, the addition to the visit_num should be 2 to balance the exploration and exploitation
                    best_child.visit_num += 1.0

                    # use the llm to get the action
                    llm_imagine_results = self.use_llm_to_imagine(state_desp, forbidden_actions, timestep=t)
                    if llm_imagine_results[0] is False:
                        # there are several cases that the imagination fails:
                        imagine_info = llm_imagine_results[1]
                        assert imagine_info in ['failure', 'exhausted', 'action_failed', 'prediction_failed'], \
                            f"imagine_info must be one of ['failure', 'exhausted', 'action_failed', 'prediction_failed'], got {imagine_info}"
                        if imagine_info == 'failure':
                            # if the imagination reaches the failure terminal state
                            # or the agent fails to generate a valid action or a valid prediction
                            failed = True
                        elif imagine_info == 'exhausted':
                            # if the state node has children, then the algorithm should use the memory
                            action, best_child = state_node.get_best(with_ucb=True, without_virtual=True)
                            child = state_node.get_child(action)
                            assert child is not None, f"child is None, action: {action}"
                            next_state_desp = child.state
                            final = (preprocess_state(next_state_desp) == preprocess_state(target_state))
                        elif imagine_info in ['action_failed', 'prediction_failed']:
                            # if the agent fails to generate a valid action or a valid prediction
                            restart = True
                            self.imagination_failure += 1
                        else:
                            raise ValueError(f"imagine_info must be one of ['failure', 'exhausted', 'action_failed', 'prediction_failed'], got {imagine_info}")
                    else:
                        # if the imagination returns a valid transition
                        next_state_desp, action, final = llm_imagine_results
                        self.temp_lang_tree.build_virtual_node(next_state_desp)
                else:
                    # if the best action is the action of an existing node, then the algorithm should ues the memory
                    self.log('----- Use memory -----')
                    child = state_node.get_child(action)
                    assert child is not None, f"child is None, action: {action}"
                    next_state_desp = child.state
                    final = (preprocess_state(next_state_desp) == preprocess_state(target_state))
                    failed = (next_state_desp == preprocess_state('Failure'))

                if final or failed or restart:
                    if final:
                        # if the imagination reaches the target state
                        self.log(f"Imagination step {t+1}:\n{action}\n{next_state_desp}")
                        self.log(f"Imagination reaches the target state:\n{target_state}")
                        self.temp_lang_tree.update(transition=[next_state_desp, 'nothing', 'Success'])
                        self.temp_lang_tree.get_node(next_state_desp).visit_num += 1
                        success_node = self.temp_lang_tree.get_node('Success')
                        self.temp_lang_tree.get_node('Success').visit_num += 1
                        self.temp_lang_tree.remove_node(next_state_desp, 'action_virtual', virtual=True)
                        success_node.value = 1
                    if failed:
                        # if the imagination reaches the failure terminal state
                        self.log(f"Imagination fails at state:\n{state_desp}")
                        failure_node = self.temp_lang_tree.get_node('Failure')
                        if failure_node is not None:
                            failure_node.value = -1
                            failure_node.visit_num += 1
                    if restart:
                        # if the imagination reaches the action failure or prediction failure
                        self.log('------- Reach the action failed or prediction failed terminal state -------')
                        self.log('------- Restart the imagination -------')

                    # compute the ucb values after each imagination step
                    self.fill_temp_lang_tree(original_state_desp=original_state_desp, target_state_desp=target_state)

                    self.log('-------------- End {}th round imagination --------------\n'.format(self.round_index))
                    break
                else:
                    self.log(f"Imagination step {t+1}:\n{action}\n{next_state_desp}")
                    state_desp = next_state_desp
                    self.temp_lang_tree.get_node(next_state_desp).visit_num += 1
                    t += 1

                    # compute the ucb values after each imagination step
                    self.fill_temp_lang_tree(original_state_desp=original_state_desp, target_state_desp=target_state)

                    if len(self.temp_lang_tree.nodes) > 40:
                        self.log('The temporary language tree is too large. The imagination process is terminated.')
                        break

            # summarize the data for this round
            stop_flag = self._summarize_this_round(
                original_state_desp=original_state_desp, target_state_desp=target_state, interrupt_flag=interrupt_flag
            )

    def use_llm_to_imagine(self, state_desp, forbidden_actions, timestep=None):
        transition, final = None, False
        imagine_results = self.imagine_trajectories(
            state_desp=state_desp, forbidden_actions=forbidden_actions, timestep=timestep
        )
        imagine_info = imagine_results['imagine_info']
        assert imagine_info in ['failure', 'exhausted', 'action_failed', 'prediction_failed', 'success'], \
            f"imagine_info must be one of ['failure', 'exhausted', 'action_failed', 'prediction_failed', 'success'], got {imagine_info}"

        if imagine_info == 'success':
            # if the imagination returns a valid transition
            transition = imagine_results['transition']
            final = imagine_results['final']
            imagine_info = 'success'

        failed = (imagine_info in ['failure', 'exhausted', 'action_failed', 'prediction_failed'])
        if failed:
            return False, imagine_info
        else:
            assert transition is not None, f"transition is None, imagine_results: {imagine_results}"
            next_state_desp = transition[-1]
            return next_state_desp, transition[1], final

    def get_llm_action(self, state_desp: str = None, feedbacks: list = [], forbidden_actions: list = [], timestep=None,
                       trial_id=None):
        target_state = self.env.target_state_desp
        policy_prompt = self.get_role_prompt('policy', state_desp, None, target_state, info={
            'feedbacks': feedbacks, 'forbidden_actions': forbidden_actions
        })
        response = self.new_query_once(policy_prompt, use_stream=True, info={
            'role': 'policy', 'timestep': timestep, 'id': trial_id, 'property': 'main'
        })

        action = self._extract_action(response, info={
            'role': 'policy', 'timestep': timestep, 'id': trial_id, 'property': 'extract_action'
        })

        return action, response

    def imagine_trajectories(self, state_desp: str = None, forbidden_actions: list = [], timestep=None):
        # input: state_desp, forbidden_actions
        # output:
        #  - if the state is a failure state, return None
        #  - if llm fails to generate a valid action, return False
        #  - if the transition exists, return transition
        target_state = self.env.target_state_desp

        action, feedbacks = None, []
        trial, action_success = 0, False
        while (trial < 3) and (action_success is not True):
            action, response = self.get_llm_action(state_desp, feedbacks, forbidden_actions, timestep+1, trial+1)
            action.replace('Error', 'ERROR')
            if 'ERROR' not in action:
                action_success, feedback = self.check_action(state_desp, action, timestep+1, trial+1)
                if not action_success:
                    feedbacks.append(feedback)
                    self.log('----- Re-plan the action -----')
                    if len(forbidden_actions) == 0:
                        trial += 1
                    else:
                        trial += 2
            else:
                action_success = False
                if len(forbidden_actions) == 0:
                    trial += 2
                else:
                    trial = 3

        if not action_success:
            if len(forbidden_actions) == 0:
                if 'ERROR' in action:
                    # if there is no forbidden action but the policy output ERROR as the action
                    # then the state reaches a failure state
                    self.temp_lang_tree.update(transition=[state_desp, 'ERROR', 'Failure'])
                    return {'transition': None, 'final': False, 'imagine_info': 'failure'}
                else:
                    # if there is no forbidden action and the policy try several times to generate a valid action but fails
                    # then the algorithm reaches a action failed state, and the algorithm should end without any treatment
                    return {'transition': None, 'final': False, 'imagine_info': 'action_failed'}
            else:
                # if there are forbidden actions and the policy cannot generate a valid action (try several times or ERROR action)
                # then the algorithm should use the memory
                return {'transition': None, 'final': False, 'imagine_info': 'exhausted'}

        action = preprocess_action(action)

        # after this step, we assume that the action is valid

        final, transition_success = False, False
        state_node = self.lang_tree.get_node(state_desp)
        transition_exists, transition = False, None
        if state_node is not None:
            if len(state_node.children) > 0:
                for child in state_node.children:
                    if action == child['action']:
                        transition = [state_desp, action, child['child'].state]
                        transition_exists = True
                        self.log('Transition exists. Use the existing transition.\n')
                        self.temp_memory_ratio['memory_count'] += 1
                        transition_success = True
                        break

        if not transition_exists:
            transition_success, transition = self.imagine_transition(state_desp, action, timestep)
            if transition_success is False:
                self.log('[Action]\n {}'.format(action) + 'Failed to imagine the transition. \n')
            else:
                self.lang_tree.update(transition)
                self.temp_memory_ratio['llm_count'] += 1

        if transition_success:
            self.temp_lang_tree.update(transition)
            if self.lang_tree.get_node(transition[-1]).state == preprocess_state(target_state):
                final = True
            return {'transition': transition, 'final': final, 'imagine_info': 'success'}
        else:
            return {'transition': None, 'final': False, 'imagine_info': 'prediction_failed'}

    def check_prediction(self, state: str, action: str, prediction: str, timestep=None, trial_id=None):
        self.log('----- Check the prediction -----')
        checker_prompt = self.get_role_prompt('prediction_checker', state, action, next_state=prediction)

        response = self.new_query_once(checker_prompt, use_stream=True, info={
            'role': 'prediction_checker', 'timestep': timestep, 'id': trial_id, 'property': 'main'
        })
        feedback_prompt = f"""Here is the feedback from a prediction checker. 
[--- Begin ---]
{response}
[--- End ---]
Please read the feedback carefully and help to summarize it. Please provide the feedback in a more concise manner. 
If the feedback says that the prediction is correct, please output [VALID]; otherwise, please output [ERROR] and provide the reason. """
        feedback = self.new_query_once(feedback_prompt, use_stream=True, info={
            'role': 'prediction_checker', 'timestep': timestep, 'id': trial_id, 'property': 'feedback'
        })
        feedback = feedback.replace('Correct', 'CORRECT')
        feedback = feedback.replace('Incorrect', 'INCORRECT')
        feedback = feedback.replace('Invalid', 'ERROR')
        feedback = feedback.replace('invalid', 'ERROR')
        feedback = feedback.replace('INVALID', 'ERROR')
        feedback = feedback.replace('Valid', 'VALID')
        feedback = feedback.replace('valid', 'VALID')

        if 'VALID' not in feedback:
            return False, feedback
        else:
            return True, response

    def check_action(self, state: str, action: str, timestep=None, trial_id=None):
        # input: state, action
        # output: if action valid, feedback
        self.log('----- Check the action -----')
        state_node = self.lang_tree.get_node(state)
        if not self.skip_check_memory:
            if state_node is not None:
                children_actions = state_node.get_children_actions(without_virtual=True)
                if preprocess_action(action) in children_actions:
                    self.log('The action is valid because it is in the full memory.')
                    return True, None

        action_checker_prompt = self.get_role_prompt('action_checker', state, action)
        action_checker_response = self.new_query_once(action_checker_prompt, use_stream=True, info={
            'role': 'action_checker', 'timestep': timestep, 'id': trial_id, 'property': 'main'
        })

        feedback_prompt = f"""Here is the feedback from an action checker. 
[--- Begin ---]
{action_checker_response}
[--- End ---]
Please read the feedback carefully and help to summarize it. Please provide the feedback in a more concise manner. 
If the feedback says that the action is valid, please output [VALID]; otherwise, please output [ERROR] and provide the reason. """
        feedback = self.new_query_once(feedback_prompt, use_stream=True, info={
            'role': 'policy', 'timestep': timestep, 'id': trial_id, 'property': 'feedback'
        })
        feedback = feedback.replace('Invalid', 'ERROR')
        feedback = feedback.replace('invalid', 'ERROR')
        feedback = feedback.replace('Valid', 'VALID')
        feedback = feedback.replace('valid', 'VALID')

        if 'VALID' not in feedback or 'ERROR' in feedback:
            return False, {'feedback': feedback, 'action': action}
        else:
            return True, None

    def _process_prediction_feedback(self, feedback: str):
        if feedback is not None:
            prediction_feedback = '[Prediction Feedback]\nI have send your prediction to a GPT-based checker to check whether your prediction is correct. The checker said that your prediction is incorrect: ' + feedback
            prediction_feedback += 'The action has been checked before so that you do not need to check it again. '
        else:
            prediction_feedback = ''

        return prediction_feedback

    def imagine_transition(self, state: str, action: str, timestep=None):
        # input: state, action
        # output:
        #  - if the transition exists, return [state, action, next_state]
        #  - if the transition does not exist, return [False, None, feedback]
        self.log('----- Imagine the transition -----')
        predictor_prompt = self.get_role_prompt('predictor', state, action, next_state=None, info={'feedbacks': ''})
        predictor_response = self.new_query_once(predictor_prompt, use_stream=True, info={
            'role': 'world_model', 'timestep': timestep + 1, 'id': 1, 'property': 'main'
        })

        predicted_next_state = self._extract_state(predictor_response)
        predicted_next_state = preprocess_state(predicted_next_state)

        prediction_passed, failed = False, False
        trial = 0
        while (prediction_passed is not True) and (trial < 2):
            if 'ERROR' in predicted_next_state:
                prediction_passed, feedback = False, 'The prediction is invalid.'
            else:
                prediction_passed, feedback = self.check_prediction(state, action, predicted_next_state, timestep+1, trial+1)

            if not prediction_passed:
                self.log('----- Re-predict the next state -----')
                trial += 1
                predictor_feedback = self._process_prediction_feedback(feedback)
                predictor_prompt = self.get_role_prompt('predictor', state, action, next_state=None, info={'feedbacks': predictor_feedback})
                predictor_response = self.new_query_once(predictor_prompt, use_stream=True, info={
                    'role': 'world_model', 'timestep': timestep + 1, 'id': trial + 1, 'property': 'main'
                })
                predicted_next_state = self._extract_state(predictor_response)
                predicted_next_state = preprocess_state(predicted_next_state)

        if trial == 2 and prediction_passed is False:
            return False, None
        else:
            return True, [state, action, predicted_next_state]
