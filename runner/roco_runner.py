import copy
import os
import json
import random
import pickle
import logging
from glob import glob
import os.path as path
from copy import deepcopy
from datetime import datetime
from typing import List, Tuple, Dict, Union, Optional, Any
from rocobench import PlannedPathPolicy, LLMPathPlan, MultiArmRRT
from prompting import LLMResponseParser, FeedbackManager, ModelPrompter
from rocobench.envs import SortOneBlockTask, MakeSandwichTask, MujocoSimEnv, SimRobot, visualize_voxel_scene
from rocobench.dialog_envs import SortOneBlockTask as SortOneBlockDialogTask
from rocobench.dialog_envs import MakeSandwichTask as MakeSandwichDialogTask
from prompting.central_prompter import DialogModelPrompter


TASK_NAME_MAP = {
    "sort_central": SortOneBlockTask,
    'sort_dialog': SortOneBlockDialogTask,
    "sandwich_central": MakeSandwichTask,
    'sandwich_dialog': MakeSandwichDialogTask,
}


class LLMRunner:
    def __init__(
            self,
            env: MujocoSimEnv,
            robots: Dict[str, SimRobot],
            max_runner_steps: int = 50,
            video_format: str = "mp4",
            epoch_id: int = 0,
            num_runs: int = 1,
            verbose: bool = False,
            np_seed: int = 0,
            start_seed: int = 0,
            run_name: str = "run",
            data_dir: str = "data",
            overwrite: bool = False,
            llm_output_mode="action_only",  # "action_only" or "action_and_path"
            llm_comm_mode="chat",
            llm_num_replans=1,
            give_env_feedback=True,
            skip_display=False,
            policy_kwargs: Dict[str, Any] = dict(control_freq=50),
            direct_waypoints: int = 0,
            max_failed_waypoints: int = 0,
            debug_mode: bool = False,
            split_parsed_plans: bool = False,
            use_history: bool = False,
            use_feedback: bool = False,
            temperature: float = 0.0,
            llm_source: str = "gpt4",
            tree_load: bool = False,
            skip_check_memory: bool = False
    ):
        self.env = env
        self.env.reset()
        self.robots = robots
        self.robot_agent_names = list(robots.keys())  # ['Alice', etc.]
        self.data_dir = data_dir
        self.run_name = run_name

        # NOTE: the run directory is data/env_name
        if epoch_id == 0:
            run_dir = os.path.join(self.data_dir, self.env.name)
        else:
            run_dir = os.path.join(self.data_dir, self.env.name + '_{}'.format(epoch_id))

        os.makedirs(run_dir, exist_ok=overwrite)
        self.run_dir = run_dir
        self.verbose = verbose
        self.np_seed = np_seed
        self.start_seed = start_seed
        self.num_runs = num_runs
        self.overwrite = overwrite
        self.direct_waypoints = direct_waypoints
        self.max_failed_waypoints = max_failed_waypoints
        self.max_runner_steps = max_runner_steps
        self.give_env_feedback = give_env_feedback
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.tree_load = tree_load
        self.skip_check_memory = skip_check_memory

        self.llm_output_mode = llm_output_mode
        self.debug_mode = debug_mode  # useful for debug

        self.llm_num_replans = llm_num_replans
        self.llm_comm_mode = llm_comm_mode
        self.response_keywords = ['NAME', 'ACTION']
        if llm_output_mode == "action_and_path":
            self.response_keywords.append('PATH')
        self.planner = MultiArmRRT(
            self.env.physics,
            robots=robots,
            graspable_object_names=self.env.get_graspable_objects(),
            allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
        )
        self.episode_id = 0
        self.policy_kwargs = policy_kwargs
        self.video_format = video_format
        self.skip_display = skip_display
        self.split_parsed_plans = split_parsed_plans
        self.temperature = temperature
        self.parser = LLMResponseParser(
            self.env,
            llm_output_mode,
            self.env.robot_name_map,
            self.response_keywords,
            self.direct_waypoints,
            use_prepick=self.env.use_prepick,
            use_preplace=self.env.use_preplace,  # NOTE: should be custom defined in each task env
            split_parsed_plans=False,  # self.split_parsed_plans,
        )
        self.feedback_manager = FeedbackManager(
            env=self.env,
            planner=self.planner,
            llm_output_mode=self.llm_output_mode,
            robot_name_map=self.env.robot_name_map,
            step_std_threshold=self.env.waypoint_std_threshold,
            max_failed_waypoints=self.max_failed_waypoints,
        )
        if 'dialog' in self.env.name:
            self.prompter = DialogModelPrompter(
                env=self.env,
                epoch_id=epoch_id,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=3000,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
                load=self.tree_load,
                skip_check_memory=skip_check_memory
            )
        else:
            self.prompter = ModelPrompter(
                env=self.env,
                parser=self.parser,
                feedback_manager=self.feedback_manager,
                max_tokens=3000,
                debug_mode=self.debug_mode,
                robot_name_map=self.env.robot_name_map,
                max_calls_per_round=10,
                use_waypoints=(self.llm_output_mode == "action_and_path"),
                use_history=self.use_history,
                use_feedback=self.use_feedback,
                num_replans=self.llm_num_replans,
                temperature=self.temperature,
                llm_source=llm_source,
                load=self.tree_load
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

        obs = self.env.reset(case_id=case_id, reload=True)
        self.env.get_target_prompt(obs=obs)

        return obs

    def display_plan(self, plan: LLMPathPlan, save_name="vis_plan", save_dir=None):
        """ Display the plan in the open3d viewer """
        env = deepcopy(self.env)
        env.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
        env.physics.forward()
        env.render_point_cloud = True
        obs = env.get_obs()
        path_ls = plan.path_3d_list
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}.jpg")
        visualize_voxel_scene(
            obs.scene,
            path_pts=path_ls,
            save_img=(save_dir is not None),
            img_path=save_path
        )

    def roco_env_step(self, llm_plan):
        obs = self.env.get_obs()
        reward, done, info = 0, False, {}
        for i, plan in enumerate(llm_plan):
            policy = PlannedPathPolicy(
                physics=self.env.physics,
                robots=self.robots,
                path_plan=plan,
                graspable_object_names=self.env.get_graspable_objects(),
                allowed_collision_pairs=self.env.get_allowed_collision_pairs(),
                plan_splitted=self.split_parsed_plans,
                **self.policy_kwargs,
            )

            num_sim_steps = 0
            plan_success, reason = policy.plan(self.env)
            if plan_success:
                while not policy.plan_exhausted:
                    sim_action = policy.act(obs, self.env.physics)
                    obs, reward, done, info = self.env.step(sim_action, verbose=False)
                    num_sim_steps += 1

        try:
            vid_name = path.join(self.prompter.env_round_save_dir, f"replay_video.mp4")
            self.env.export_render_to_video(vid_name, out_type=self.video_format, fps=50)
        except:
            pass

        return obs, reward, done, info

    def test_one_episode(self, initial_obs):
        # the initial_obs is the initial observation of the current testing case
        # At first, we need to check if the target state is reachable
        initial_state = self.env.get_state_prompt(initial_obs)
        path_exist = self.prompter.check_path_exist(initial_state, self.env.target_state_desp)

        env_round_save_dir = path.join(self.prompter.interact_save_dir, f"round_{self.episode_id}")
        self.prompter.set_env_round_dir(env_round_save_dir)

        self.prompter.log(f"-------------- Start testing case {self.episode_id} --------------", if_test=True)
        self.prompter.log(f"Initial state: {initial_state}", if_test=True)
        self.prompter.log(f"Target state: {self.env.target_state_desp}", if_test=True)

        IK_failed = False
        step, failed = 0, False
        state = initial_state
        obs = initial_obs
        while step < self.max_runner_steps:
            self.prompter.log(f"------ Step {step} ------", if_test=True)
            self.prompter.fill_temp_lang_tree(state, self.env.target_state_desp)
            ready_to_execute, current_llm_plan, path_exist = self.prompter.decision_with_model(
                obs, path_exist=path_exist
            )
            if ready_to_execute is None and current_llm_plan is None and path_exist is None:
                self.prompter.log("The run failed due to IK failure. Restart the interaction test.", if_test=True)
                IK_failed = True
                break

            if not ready_to_execute:
                # the llm agent fails to find a valid action
                failed = True
                break

            # obs, reward, done, info = self.env.step(current_llm_plan, verbose=False)
            obs, reward, done, info = self.roco_env_step(current_llm_plan)
            state = self.env.get_state_prompt(obs)

            # log the transition
            self.prompter.log(f"Action: \n{current_llm_plan[0].parsed_proposal}", if_test=True)
            self.prompter.log(f"State: \n{state}", if_test=True)

            step += 1

            if done:
                self.episode_id += 1
                break

        self.prompter.log("Run finished after {} timesteps".format(step), if_test=True)

        if IK_failed:
            return None

        if failed:
            self.prompter.log("The run failed", if_test=True)
            return -1

        return step

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
        with open(path.join(self.prompter.case_dir, "Task.txt"), "a") as f:
            f.write(f"Original state:\n{state}\n")
            f.write(f"Target state:\n{self.env.target_state_desp}\n")
            f.write(f"Initial Information:\n{str(self.env.get_initial_information())}\n")

        print('---------- Task Information ----------')
        print(f"Original state:\n{state}\n")
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

            end_step = None
            IK_failed = True
            while IK_failed:
                # start an episode of interaction with the environment
                end_step = self.test_one_episode(initial_obs=obs)
                if end_step is None:
                    IK_failed = True
                else:
                    IK_failed = False

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
