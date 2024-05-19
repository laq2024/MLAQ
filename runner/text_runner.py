import copy
import os
import json
import random
from copy import deepcopy
from glob import glob
import os.path as path
from datetime import datetime
from prompting import TextFeedbackManager, SinglePrompter


class LLMRunner:
    def __init__(
            self, env, max_runner_steps: int = 50, num_runs: int = 1, np_seed: int = 0,
            start_seed: int = 0, run_name: str = "run", data_dir: str = "data", overwrite: bool = False,
            llm_num_replans=1, give_env_feedback=True, skip_display=False, debug_mode: bool = False,
            use_history: bool = False, use_feedback: bool = False, temperature: float = 0.0, llm_source: str = "gpt4",
            tree_load: bool = False, skip_check_memory: bool = False
    ):
        self.env = env
        self.env.reset(case_id=0)
        self.data_dir = data_dir
        self.run_name = run_name
        self.episode_id = 0
        self.skip_check_memory = skip_check_memory

        # NOTE: the run directory is data/env_name
        run_dir = os.path.join(self.data_dir, self.env.name)
        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.tree_load = tree_load

        self.debug_mode = debug_mode  # useful for debug

        self.llm_num_replans = llm_num_replans
        self.response_keywords = ['NAME', 'ACTION']
        self.skip_display = skip_display
        self.temperature = temperature
        self.feedback_manager = TextFeedbackManager(env=self.env)
        self.prompter = SinglePrompter(
            env=self.env,
            parser=None,
            feedback_manager=self.feedback_manager,
            max_tokens=3000,
            debug_mode=self.debug_mode,
            robot_name_map=self.env.robot_name_map,
            max_calls_per_round=10,
            use_waypoints=False,
            use_history=self.use_history,
            use_feedback=self.use_feedback,
            num_replans=self.llm_num_replans,
            temperature=self.temperature,
            llm_source=llm_source,
            load=self.tree_load,
            skip_check_memory=self.skip_check_memory
        )

    def get_case_end(self, end_step: int):
        # if the optimal steps are reached, we consider the case ends
        if end_step <= self.env.optimal_steps:
            return True, 'success'

        # if the interaction fails for more than 3 times (end_step = -1), we consider the case ends
        if end_step == -1:
            return True, 'failure'

        # if the language tree is too large, we consider the case ends
        if len(self.prompter.temp_lang_tree.nodes) > 40:
            return True, 'failure'

        # if the episode id is larger than 3, we consider the case ends
        if self.episode_id >= 3:
            return True, 'failure'

        return False, None

    def reset_env(self, case_id):
        # seed = case_id + start_id
        seed = 0
        self.env.seed(np_seed=seed)
        random.seed(seed)

        obs = self.env.reset(case_id=case_id)
        self.env.get_target_prompt(obs=obs)

        return obs

    def one_run_with_model(self, case_id):
        """ run one case of the task """
        # case_result is a dictionary that records the success and converge tokens
        case_result = {
            'success_token': None, 'success_memory_ratio': None,
            'converge_token': None, 'converge_memory_ratio': None,
            'end_step': None
        }
        obs = self.reset_env(case_id)
        state = self.env.get_state_prompt(obs)
        self.prompter.original_state_desp = copy.deepcopy(state)
        self.prompter.initial_temp_lang_tree(
            initial_obs=obs, target_state_desp=self.env.target_state_desp
        )

         # log the task information at the beginning of the whole imagination process
        with open(path.join(self.prompter.case_dir, "Task.txt"), "a") as f:
            f.write(f"Original state:\n{obs}\n")
            f.write(f"Target state:\n{self.env.target_state_desp}\n")
            f.write(f"Initial Information:\n{str(self.env.get_initial_information())}\n")

        print('---------- Task Information ----------')
        print(f"Original state:\n{obs}\n")
        print(f"Target state:\n{self.env.target_state_desp}\n")
        print(f"Initial Information:\n{str(self.env.get_initial_information())}\n")
        print('-------------------------------------')

        first_success, case_end = False, False
        while not case_end:
            obs = self.reset_env(case_id)

            # use the prompter to sample trajectories in the imagination space based on the LLM
            self.prompter.round_reset()
            self.prompter.imagine_with_ucb(obs=obs, horizon=12, interrupt_flag=False)

            if self.prompter.imagination_failure >= 3:
                print("Imagination failure. Pass to the next case.")
                break
            # start an episode of interaction with the environment
            end_step = self.test_one_episode(initial_obs=obs)

            # check if the case should be terminated
            case_end, result = self.get_case_end(end_step)

            if result == 'success':
                # if the case ends with reaching the optimal steps, we record the token and memory ratio
                assert case_end is True, "The case should end if it succeeds"
                case_result['converge_token'], case_result['converge_memory_ratio'] = self.prompter.get_summary()
                llm_counts = case_result['converge_memory_ratio']['llm_count']
                memory_counts = case_result['converge_memory_ratio']['memory_count']
                if len(llm_counts) == 1 and (llm_counts[0] + memory_counts[0]) == 0:
                    case_result['converge_memory_ratio']['memory_count'] = [end_step]
                if case_result['success_memory_ratio'] is None:
                    case_result['success_memory_ratio'] = case_result['converge_memory_ratio']
                    case_result['success_token'] = case_result['converge_token']
            elif result is None:
                if first_success is False:
                    # if the test first reaches the target state, we record the token and memory ratio
                    first_success = True
                    case_result['success_token'], case_result['success_memory_ratio'] = self.prompter.get_summary()
                    if len(case_result['success_memory_ratio']) == 1 and case_result['success_memory_ratio'][0] == 0:
                        case_result['success_memory_ratio'] = [end_step]
            elif result == 'failure':
                assert case_end is True, "The case should end if it fails"
            else:
                raise ValueError(f"Unknown result {result}")

        case_result['end_step'] = end_step
        with open(path.join(self.prompter.case_dir, "case_result.json"), "w") as f:
            json.dump(case_result, f)

    def test_one_episode(self, initial_obs):
        # the initial_obs is the initial observation of the current testing case
        # At first, we need to check if the target state is reachable
        path_exist = self.prompter.check_path_exist(initial_obs, self.env.target_state_desp)

        env_round_save_dir = path.join(self.prompter.interact_save_dir, f"round_{self.episode_id}")
        self.prompter.set_env_round_dir(env_round_save_dir)

        self.prompter.log(f"-------------- Start testing case {self.episode_id} --------------", if_test=True)
        self.prompter.log(f"Initial state: {initial_obs}", if_test=True)
        self.prompter.log(f"Target state: {self.env.target_state_desp}", if_test=True)

        step, failed = 0, False
        obs = initial_obs
        while step < self.max_runner_steps:
            self.prompter.log(f"------ Step {step} ------", if_test=True)
            self.prompter.fill_temp_lang_tree(obs, self.env.target_state_desp)
            ready_to_execute, current_llm_plan, path_exist = self.prompter.decision_with_model(
                obs, path_exist=path_exist
            )
            if not ready_to_execute:
                # the llm agent fails to find a valid action
                failed = True
                break

            obs, reward, done, info = self.env.step(current_llm_plan, verbose=False)

            # log the transition
            self.prompter.log(f"Action: \n{current_llm_plan}", if_test=True)
            self.prompter.log(f"State: \n{obs}", if_test=True)

            step += 1

            if done:
                self.episode_id += 1
                break

        self.prompter.log("Run finished after {} timesteps".format(step), if_test=True)

        if failed:
            self.prompter.log("The run failed", if_test=True)
            return -1

        return step

    def run(self, args):
        # if we want to start from a specific run id, we can set the start_id manually
        start_id = args.start_run_id

        save_dir = os.path.join(self.run_dir, f"run_{start_id}")
        os.makedirs(save_dir, exist_ok=self.overwrite)

        print(f"==== Run {start_id} starts ====")
        # save args into a json file
        args_dict = vars(args)
        args_dict["env"] = self.env.__class__.__name__
        timestamp = datetime.now().strftime("%Y%m_%H%M")
        fname = os.path.join(save_dir, f"args_{timestamp}.json")

        os.makedirs(os.path.dirname(fname), exist_ok=True)
        json.dump(args_dict, open(fname, "w"), indent=2)

        start_case_id = args.start_case_id
        total_test_number = min(self.env.case_number, 30)
        for case_id in range(start_case_id, total_test_number):
            # set the saving directories for the prompter
            # for this run, we have a main directory named run_{start_id}
            # then we have subdirectories for each simulation round and each test round (interact with the environment)
            case_dir = os.path.join(self.run_dir, f"run_{start_id}", f"case_{case_id}")
            self.prompter.set_case_dir(case_dir)
            self.prompter.lang_tree.refine_tree()

            print(f"============ Case {case_id} starts ============")
            self.episode_id = 0
            self.prompter.case_reset()
            self.one_run_with_model(case_id)
            print(f"============ Case {case_id} ends ============\n\n")
