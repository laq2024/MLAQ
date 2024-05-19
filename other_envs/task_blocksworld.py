import re
import yaml
import json
import os.path as path
import numpy as np
from typing import List
import random, copy
import os.path as path
from tarski.io import PDDLReader


BLOCKSWORLD_ALL_OBJECTS = [
    "blue block",
    "orange block",
    "red block",
    "yellow block",
]

BLOCKSWORLD_TASK_CONTEXT = """ 
4 blocks on the table. You need to rearrange them into stacks.
"""

BLOCKSWORLD_DIALOG_PROMPT = ""

BLOCKSWORLD_CHAT_PROMPT = ""

BLOCKSWORLD_PLAN_PROMPT = """
Think step-by-step and reason about the best strategy to achieve the goal. Carefully consider Environment Feedback and Scene Description.
At each round, plan **exactly** one ACTION. 
"""

BLOCKSWORLD_ACTION_SPACE = """[Action Space Definition]
Detail the action space with the stipulated actions: PICK UP, UNSTACK, PUT DOWN, STACK ON.
- PICK UP <object>: Execute if hand is empty, and <object> is on table and no block is on it. Pick up <object> from table. After execution, you will be holding the block.
- UNSTACK <object>: Execute if hand is empty, <object> is on another block and no block is on it. UNSTACK <object> from another block. After execution, you will be holding the block.
- PUT DOWN <object>: Execute if hand is holding <object>. Put down <object> on table. After execution, your hand will be empty.
- STACK <object> ON <target>: Execute if hand is holding <object>, and no block is on <target>. Stack <object> on the top of <target>. After execution, your hand will be empty.

[Action Template]
EXECUTE
<action>

[Action Output Instructions]
1. Commence the output with 'EXECUTE\n'. Follow with one action. 
2. Don't add extra symbols at the end of each line, such as periods.

[Action Planning Instructions]
At first, you should first repeat the state as follows: [Repeat current State] and [Repeat target State].
  + [Repeat current State]: [red block: [], blue block: [], orange block: [], yellow block: []; hand: []]
  + [Repeat target State]: [red block: [], blue block: [], orange block: [], yellow block: []; hand: []]
[Constraints]: You can only do one action at a time. Do not choose the action listed in the [Forbidden Actions].
Please output your thinking process step-by-step. You have to follow these steps one by one to plan the [Action]: [Action Plan] and [Pre Action Conclusion].
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions to not omit any steps.
a. [Action Plan] Plan the [Action] of the agent step by step and list the thinking process.
  + Follow the [Action Output Instructions] and [Detailed Constraints for Action] to plan the [Action] of the agent. You can also take the following tips (Tips 1 and Tips 2) into consideration.
  + [Tips 1]: If you want to PICK or UNSTACK a block with other blocks on it, you should first UNSTACK the blocks on top of it.
  + [Tips 2]: Do not forget that when you are holding a block, you can only PUT DOWN or STACK ON it and you cannot PICK UP or UNSTACK other blocks.
b. [Pre Action Conclude] Conclude the [Action] according to the [Action Planning] in the format of [Detailed Constraints for Action]. Conclusion should be in the order [hand; red block, blue block, orange block, yellow block].

[Detailed Constraints for Action]
[Detailed Constraints] Now, you should follow the constraints one-by-one and step-by-step to check if the action is correct: [Basic Check] and [Action Check].
You have to follow these constraints strictly and do not have your own understanding of the constraints.
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content. You should also fill in the <object> and <target> with the correct block names when needed.
1. [Basic Check] Please output "[Basic Check]" and do as follows:
  + List the line of the actions: []. There must be one line of actions. Count the number: []. Check if it is equal to one: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + List the action: []. Actions asides from these are not allowed. Check if the action is one of "PICK UP", "UNSTACK", "PUT DOWN" and "STACK ON": [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
2. [Action Check] If action is "PICK UP", go to [PICK UP Check]; if action is "UNSTACK", go to [UNSTACK Check]; if action is "PUT DOWN", go to [PUT DOWN Check]; if action is "STACK ON", go to [STACK ON Check].
  + [PICK UP Check] Please output "[PICK UP Check]" and do as follows:
    a. List the state of your hand: []. Your hand must be empty. Check if your hand is empty: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    b. List the <object> of PICK UP: []. <object>'s state must be 'on table'. List the state of <object>: []. Check if the <object> is 'on table': [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    c. List the <object> of PICK UP: []. No <object>'s state can be 'on <object>'. In the repeated current state, list the blocks whose state is 'on <target>': []. List the state of the above block: [], check again if the state of the above block is 'on <target>': []. If check is passed, count the number of above blocks: []. Check if the number is zero: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + [UNSTACK Check] Please output "[UNSTACK Check]" and do as follows:
    a. List the state of your hand: []. Your hand must be empty. Check if your hand is empty: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    b. List the <object> of UNSTACK: []. <object>'s state must be on another block. List the state of <object>: []. Check if the <object> is 'on <another block>': [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    c. List the <object> of UNSTACK: []. No <object>'s state can be 'on <object>'. Repeat current State: []. In the repeated current state, list the blocks whose state is 'on <target>': []. List the state of the above block: [], check again if the state of the above block is 'on <target>': []. If check is passed, count the number of above blocks: []. Check if the number is zero: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + [PUT DOWN Check] Please output "[PUT DOWN Check]" and do as follows:
    a. List the state of your hand: []. Your hand cannot be empty. Check if your hand is holding <object>: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + [STACK ON Check] Please output "[STACK ON Check]" and do as follows:
    a. List the state of your hand: []. Your hand cannot be empty. Check if your hand is holding <object>: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    b. List the <object> of STACK: []. <object> must be in your hand. List the state of <object>: [], and the state of your hand: []. Check if the <object> is in your hand: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
    c. List the <target> of ON: []. No <object>'s state can be 'on <target>'. Repeat current State: []. In the repeated current state, list the blocks whose state is 'on <target>': []. List the state of the above block: [], check again if the state of the above block is 'on <target>': []. If check is passed, count the number of above blocks: []. Check if the number is zero: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].

[Example Actions]
Example#1: EXECUTE\nPICK UP orange block.\n
Example#2: EXECUTE\nSTACK blue block ON red block.\n
"""

BLOCKSWORLD_STATE_SPACE = """[State Space Definition]
Define the state of the scenarios: blocks and hand.
1. [Hand State]: The state of the hand. It can be "Empty" or "Holding <object>", where <object> is a block.
2. [Block States]: The state of each block. It can only be one of ["on <object>", "on table", "in hand"]. The blocks should be listed in this order: blue block, orange block, red block, yellow block.

[State Template]
[State]
[Hand State]
<hand state>
[Block States]
blue block: <block state>
orange block: <block state>
red block: <block state>
yellow block: <block state>

[Detailed Constraints for State]
The following paragraphs are the detailed constraints for State. There are three main <check items>: [Title Check], [Hand State Check], and [Block States Check].
1. [Title Check] A header titled [State].
2. [Hand State Check] A header titled [Hand State]. Then, list the Hand State.
  + Hand state must be one of ["Empty", "Holding <object>"], where <object> is a block (e.g. "Holding blue block").
3. [Block States Check] A header titled [Block States]. Then, list the states of each block.
  + Each line represents the state of one block item.
  + The block items must be listed in this order: blue block, orange block, red block, yellow block.
  + The <block state> specifies the state of the block. It can only be "on <object>", "on table" or "in hand" (e.g. "blue block: on table", "orange block: on red block", "red block: in hand").

[Example States]
Example#1:
[State]
[Hand State]
Empty
[Block States]
blue block: on red block
orange block: on blue block
red block: on table
yellow block: on table
"""

prediction_example = f"""[Prediction Example]
Example#1: 
[State]
[Hand State]
Empty
[Block States]
blue block: on orange block
orange block: on table
red block: on table
yellow block: on table

[Action]
EXECUTE
UNSTACK yellow block

[Prediction]
[State]
[Hand State]
Holding yellow block
[Block States]
blue block: on orange block
orange block: on table
red block: on table
yellow block: in hand
"""

BLOCKSWORLD_PREDICTING_PROMPT = f"""[Predicting Instruction]
You will be provided with the [State] and the [Action] of the agent. You should think step by step to output the [Prediction] of the next [State] based on the given [State] and [Action]. The format of the [Prediction] should follow the [Detailed Constraints for State].
Please output your thinking process step-by-step by following theses steps:
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content.
1. [Interaction Item Pool]: initialize the pool of the blocks that the agent is interacting with. It is a empty list at the beginning: pool={{}}.
2. [Action Forward Rule]: List the action: [], the state in the [Prediction] is changed by the [Action]. Follow these steps to predict the [Prediction]:
  + If the action is "PICK UP" or "UNSTACK", list the <object> of action: []. The state of <object> in the [Prediction] should be changed to "in hand". The state of your hand in the [Prediction] should be changed to "Holding <object>". Add <object> to the pool: pool=[].
  + If the action is "PUT DOWN", list the <object> of action: []. The state of <object> in the [Prediction] should be changed to "on table". The state of your hand in the [Prediction] should be changed to "Empty". Add <object> to the pool: pool=[].
  + If the action is "STACK ON", list the <object> of action: [], and <target> of action: []. The state of <object> in the [Prediction] should be changed to "on <target>". The state of your hand in the [Prediction] should be changed to "Empty". Add <object> to the pool: pool=[].
3. [Prediction Conclusion]: conclude the [Prediction] based on the [Action Forward Rule]. Please output the [Prediction] in the format of [State Space Definition] at the end of this part.
  + The format of the [Prediction] should follow the [Detailed Constraints for State].
  + The blocks that are not present in the pool should not change their state.
4. [Block State Consistency Check]: check if the block states are consistent with the action.
  + Repeat the full block list: [blue block, orange block, red block, yellow block].
  + Repeat the pool: [], and remove the items in the pool from the full block list: [].
  + List the state of the remaining blocks in the above list: [].
  + Repeat the [Prediction]. List the [Prediction]'s block state in the above list: []. Check if the blocks' states in the [Prediction] are consistent with the [State]: [yes/no]. If yes, the [Prediction] is correct; otherwise incorrect. Conclusion: [].
5. [Prediction Justification]: check the [Prediction] AGAIN based on the [Action Forward Rule] and the [Block State Consistency Check]. If the [Prediction] is incorrect, justify the mistake and correct it.
6. [Conclusion]: conclude your thinking process to output the [Prediction] in the format of [State Space Definition]. This part only contains the outputted [Prediction].
{prediction_example}
"""


class BlocksWorldTask:
    def __init__(self, initial_coef=0.5, target_coef=0.5):
        self.block_names = ["blue block", "orange block", "red block", "yellow block"]

        self.initial_coef, self.target_coef = initial_coef, target_coef
        self.optimal_steps = 2
        self.case_number = 0
        self.cases_list = []
        self.name = 'blocksworld'

        self.block_stacks = []  # save each block stack as a list, from bottom to top

        self.target_stacks = []  # this should have the same format as self.block_stacks
        self.target_state_desp = ""  # this should be a description for a part of self.target_stacks

        self.block_in_hand = None  # can be `None` or a block name

        # place holder
        self.physics = None
        # place holder
        self.robot_name_map = {"panda": "Bob"}

    @property
    def use_preplace(self):
        return True

    def set_optimal_steps(self, optimal_steps):
        self.optimal_steps = optimal_steps
        data_list = json.load(open('other_envs/full_data/step_{}.json'.format(self.optimal_steps), 'r'))
        domain_pddl = "other_envs/generated_domain.pddl"
        for data in data_list:
            case_reader = PDDLReader(raise_on_error=True)
            case_reader.parse_domain(domain_pddl)
            problem = case_reader.parse_instance(
                path.join('other_envs/blocksworld_cases', *data[0].split('/')[-2:])
            )
            if int(problem.name[-1]) == 4:
                self.cases_list.append(data)
                self.case_number += 1

    def parse_case(self, case_id: int):
        case_data = self.cases_list[case_id]
        assert case_data[-1] == self.optimal_steps

        cur_instance = path.join('other_envs/blocksworld_cases', *case_data[0].split('/')[-2:])
        domain_pddl = "other_envs/blocksworld_cases/generated_domain.pddl"
        case_reader = PDDLReader(raise_on_error=True)
        case_reader.parse_domain(domain_pddl)
        problem = case_reader.parse_instance(cur_instance)
        with open('other_envs/bw_config.yaml', 'r') as file:
            config_data = yaml.safe_load(file)

        init_stacks = self._get_stacks(problem)
        plan_str = case_data[1]
        plan = self._parse_plan(plan_str)

        stacks = copy.deepcopy(init_stacks)
        for action in plan:
            action_type = str(list(action.keys())[0])
            obj, target = list(action.values())[0]
            stacks = self._fast_step(stacks, action_type, obj, target)

        def parse_stacks(simple_stacks):
            block_stacks = []
            for stack in simple_stacks:
                temp_stack = []
                for stack_item in stack:
                    temp_stack.append(config_data['encoded_objects'][stack_item])

                block_stacks.append(temp_stack)

            return block_stacks

        parsed_init_stacks = parse_stacks(init_stacks)
        parsed_goal_stacks = parse_stacks(stacks)

        return parsed_init_stacks, parsed_goal_stacks

    def _parse_plan(self, plan_str):
        actions = plan_str.split("\n")
        for action in actions:
            if len(action) == 0:
                actions.remove(action)

        pattern = r'(pick-up|put-down|stack|unstack)\s(\w+)\s?(\w+)?'
        parsed_plan = []
        for action in actions:
            action_type = re.search(pattern, action).group(1)
            obj = re.search(pattern, action).group(2)
            target = re.search(pattern, action).group(3)
            parsed_plan.append({
                action_type: [obj, target]
            })

        return parsed_plan

    def _get_stacks(self, problem):
        stacks = []
        on_table_objects = []
        for atom in problem.init.as_atoms():
            if atom.symbol.name == 'ontable':
                on_table_objects.append(atom.subterms[0].name)
                stacks.append([atom.subterms[0].name])

        for index, on_table_object in enumerate(on_table_objects):
            end_loop = False
            last_object = on_table_object
            while not end_loop:
                state_exist = False
                for atom in problem.init.as_atoms():
                    if atom.symbol.name == 'on' and atom.subterms[1].name == last_object:
                        stacks[index].append(atom.subterms[0].name)
                        last_object = atom.subterms[0].name
                        state_exist = True
                        break

                if not state_exist:
                    end_loop = True

        return stacks

    def _fast_step(self, stacks, action: str, obj: str, target: str):
        if action == 'pick-up':
            for i in range(len(stacks)):
                if stacks[i][-1] == obj:
                    stacks[i].pop(-1)
            while True:
                try:
                    stacks.remove([])
                except ValueError:
                    break
        elif action == 'put-down':
            stacks += [[obj]]
        elif action == 'stack':
            for stack in stacks:
                if stack[-1] == target:
                    stack.append(obj)
                    break
        elif action == 'unstack':
            for i in range(len(stacks)):
                if stacks[i][-1] == obj:
                    stacks[i].pop(-1)
            while True:
                try:
                    stacks.remove([])
                except ValueError:
                    break
        else:
            raise ValueError(f"Unknown action: {action}")

        return stacks

    def seed(self, np_seed):
        self.random_state = np.random.RandomState(np_seed)

    def get_action_prompt(self) -> str:
        return BLOCKSWORLD_ACTION_SPACE

    def get_action_space_prompt(self) -> str:
        return BLOCKSWORLD_ACTION_SPACE

    def get_state_space_prompt(self) -> str:
        return BLOCKSWORLD_STATE_SPACE

    def get_predicting_prompt(self):
        return BLOCKSWORLD_PREDICTING_PROMPT

    def get_state_prompt(self, obs):
        state_desp = self.get_obs(block_stacks=self.block_stacks, block_in_hand=self.block_in_hand)
        return state_desp

    # place holder
    def get_sim_robots(self):
        return {"a": "a"}

    def step(self, action: str, verbose=False):
        if re.search("PICK UP (.* block)", action) is not None:
            block_name = re.search("PICK UP (.* block)", action).group(1)
            self.block_in_hand = block_name
            for i in range(len(self.block_stacks)):
                if self.block_stacks[i][-1] == block_name:
                    self.block_stacks[i].pop(-1)
            while True:
                try:
                    self.block_stacks.remove([])
                except ValueError:
                    break
        elif re.search("PUT DOWN (.* block)", action) is not None:
            block_name = re.search("PUT DOWN (.* block)", action).group(1)
            self.block_in_hand = None
            self.block_stacks += [[block_name]]
        elif re.search("STACK (.* block) ON (.* block)", action) is not None:
            re_match = re.search("STACK (.* block) ON (.* block)", action)
            block_name1, block_name2 = re_match.group(1), re_match.group(2)
            self.block_in_hand = None
            for stack in self.block_stacks:
                if stack[-1] == block_name2:
                    stack = stack.append(block_name1)
                    break
        elif re.search("UNSTACK (.* block)", action) is not None:
            block_name = re.search("UNSTACK (.* block)", action).group(1)
            self.block_in_hand = block_name
            for i in range(len(self.block_stacks)):
                if self.block_stacks[i][-1] == block_name:
                    self.block_stacks[i].pop(-1)
            while True:
                try:
                    self.block_stacks.remove([])
                except ValueError:
                    break
        next_obs = self.get_obs(block_stacks=self.block_stacks, block_in_hand=self.block_in_hand)
        reward, done = self.get_reward_done(next_obs)

        return next_obs, reward, done, {}

    def reset(self, case_id: int):
        # case_id is the index of the case in self.cases_list: 0 <= case_id < self.case_number
        assert 0 < self.target_coef <= 1
        assert 0 < self.initial_coef <= 1

        self.block_stacks = []
        self.target_stacks = []
        self.block_in_hand = None

        # initialize the blocks
        # blocks = copy.deepcopy(self.block_names)
        # random.shuffle(blocks)
        # for block in blocks:
        #     if random.random() < self.initial_coef or self.block_stacks == []:
        #         self.block_stacks.append([])
        #     self.block_stacks[-1].append(block)

        # generate the target
        # random.shuffle(blocks)
        # for block in blocks:
        #     if random.random() < self.target_coef or self.target_stacks == []:
        #         self.target_stacks.append([])
        #     self.target_stacks[-1].append(block)

        # initialize the blocks
        initial_stack, target_stacks = self.parse_case(case_id)

        self.block_stacks, self.target_stacks = initial_stack, target_stacks

        # there is no block in hand in target state
        self.target_state_desp = self.get_obs(block_stacks=self.target_stacks, block_in_hand=None)

        obs = self.get_obs(block_stacks=self.block_stacks, block_in_hand=None)

        return obs

    # changed from describe_robot_state
    def describe_hand_state(self, block_in_hand):
        if block_in_hand is not None:
            return f"Holding {block_in_hand}\n"
        else:
            return "Empty\n"

    def describe_block_state(self, block_stacks, block_name, block_in_hand):
        if block_name not in self.block_names:
            raise Exception("The block_name parameter of describe_block_state is not in self.block_names")

        block_desp = ""
        if block_name == block_in_hand:
            block_desp += f"{block_name}: in hand\n"

        for stack in block_stacks:
            if block_name in stack:
                assert len(block_desp) == 0
                index = stack.index(block_name)
                bottom_block_name = stack[index - 1] if index > 0 else "table"
                block_desp += f"{block_name}: on {bottom_block_name}\n"

        return block_desp

    def get_obs(self, block_stacks=None, block_in_hand=None):
        if block_stacks is None:
            block_stacks = self.block_stacks

        obs = "[State]\n[Hand State]\n" + self.describe_hand_state(block_in_hand)
        obs += "[Block States]\n"
        for block in self.block_names:
            obs += self.describe_block_state(block_stacks, block, block_in_hand)

        return obs

    def get_reward_done(self, obs):
        reward = 1
        done = True
        target_prompt = self.get_target_prompt(obs)
        target_prompt_constraints = target_prompt.split("\n")
        current_state_desc = obs
        for block_name in self.block_names:
            current_state_desc += self.describe_block_state(self.block_stacks, block_name, self.block_in_hand)
        for constraints in target_prompt_constraints:
            if constraints not in current_state_desc:
                reward = 0
                done = False
                break

        return reward, done

    def describe_task_context(self):
        return BLOCKSWORLD_TASK_CONTEXT

    # changed
    def get_target_prompt(self, obs):
        return self.target_state_desp

    def get_task_feedback(self, llm_plan):
        feedback = ""
        error_info = {
            'dynamics_error': False,
            'task_error': False
        }
        if re.search("PICK UP .* block", llm_plan) is not None:
            block_name = re.search("PICK UP (.* block)", llm_plan).group(1)
            if all(stack[0] != block_name for stack in self.block_stacks):
                feedback += f"{block_name} is not on table, so it cannot be PICKed UP."
            if all(stack[-1] != block_name for stack in self.block_stacks):
                feedback += f"{block_name} has other blocks on it, so it cannot be PICKed UP."
        elif re.search("UNSTACK .* block", llm_plan) is not None:
            block_name = re.search("UNSTACK (.* block)", llm_plan).group(1)
            if any(stack[0] == block_name for stack in self.block_stacks):
                feedback += f"{block_name} is on table, so it cannot be UNSTACKed."
            if all(stack[-1] != block_name for stack in self.block_stacks):
                feedback += f"{block_name} has other blocks on it, so it cannot be UNSTACKed."
        elif re.search("PUT DOWN .* block", llm_plan) is not None:
            block_name = re.search("PUT DOWN (.* block)", llm_plan).group(1)
            if block_name != self.block_in_hand:
                feedback += f"{block_name} is not in hand, so it cannot be PUT DOWN."
        elif re.search("STACK .* block ON .* block", llm_plan) is not None:
            search_result = re.search("STACK (.* block) ON (.* block)", llm_plan)
            block_name1, block_name2 = search_result.group(1), search_result.group(2)
            if block_name1 != self.block_in_hand:
                feedback += f"{block_name1} is not in hand, so it cannot be STACKed."
            if all(stack[-1] != block_name2 for stack in self.block_stacks):
                feedback += f"{block_name2} has other blocks on it, so it cannot be STACKed ON."
        else:
            feedback += "Action is not allowed. The available actions are: PICK UP <object>, UNSTACK <object>, PUT DOWN <object>, STACK <object> ON <target>."

        if len(feedback) > 0:
            error_info['dynamics_error'] = True

        return feedback, error_info

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return BLOCKSWORLD_CHAT_PROMPT

    def central_plan_prompt(self):
        return BLOCKSWORLD_PLAN_PROMPT

    def dialog_mode_prompt(self):
        return BLOCKSWORLD_DIALOG_PROMPT

    def get_task_prompt(self):
        task_prompt = f"""[Task Description]
Task: Rearrange the blocks on the table.
O
Objective: Rearrange the blocks on the table to meet all the constraints in [Target State]. Please note that if a block is not mentioned in [Target State], we have no constraint with respect to it. However, don't hold any blocks in your hand when you consider the [Target State] to be accomplished.

{self.get_target_prompt(None).replace("[State]", "[Target State]")}

Instructions:
1. You can only perform actions described in [Action Space Definition]. Please do not imagine other actions.
2. This is a Markov Decision Process (MDP). There exists an external environment interacting with the you. 

Constraints:
1. The state of the system must adhere to the prescribed [State Space Definition]! 
2. You should select one action from the provided [Action Space Definition]. 
"""
        return task_prompt

    def get_initial_information(self):
        return 'nothing'

    def get_repeat_template(self):
        state_template = '[]'
        action_template = '[]'

        return {'state': state_template, 'action': action_template}

    def get_examples(self):
        plan_example = f"""[Plan Example]
Example#1:
[Original State]
[Hand State]
Empty
[Block States]
blue block: on table
orange block: on table
red block: on table
yellow block: on red block

[Target State]
[Hand State]
Empty
[Block States]
blue block: on orange block,
orange block: on table
red block: on yellow block
yellow block: on table

[Action]
EXECUTE
UNSTACK yellow block
"""

        return {'plan': plan_example}