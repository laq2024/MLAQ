import os
import json
import openai
import logging
from datetime import datetime
import argparse

# from runner.roco_runner import LLMRunner, TASK_NAME_MAP
from runner.roco_runner import LLMRunner, TASK_NAME_MAP
from runner.text_runner import LLMRunner as TextLLMRunner
from other_envs.task_blocksworld import BlocksWorldTask


# print out logging.info
logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)

# add your OpenAI API key and the base URL here
OPENAI_KEY = str("sk-4LGEn4h1YckvuKaA288c1102D1E34170AfC3998328196c57")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
os.environ["OPENAI_BASE_URL"] = "https://chat1.plus7.plus/v1"
openai.api_based = "https://chat1.plus7.plus/v1"
openai.api_key = OPENAI_KEY


def main(args):
    if args.task in ['sort_dialog', 'sandwich_dialog', 'sort_central', 'sandwich_central']:
        assert args.task in TASK_NAME_MAP.keys(), f"Task {args.task} not supported"
        env_cl = TASK_NAME_MAP[args.task]

        render_freq = 600
        if args.control_freq == 15:
            render_freq = 1200
        elif args.control_freq == 10:
            render_freq = 2000
        elif args.control_freq == 5:
            render_freq = 3000
        env = env_cl(
            render_freq=render_freq,
            image_hw=(400, 400),
            sim_forward_steps=300,
            error_freq=30,
            error_threshold=1e-5,
            randomize_init=True,
            render_point_cloud=0,
            render_cameras=["face_panda", "face_ur5e", "teaser", ],
            one_obj_each=True,
        )
        env.set_optimal_steps(args.optimal_steps)

        robots = env.get_sim_robots()
        if not os.path.exists(args.data_dir + "/{}".format(args.task)):
            os.makedirs(args.data_dir + "/{}".format(args.task))

        runner = LLMRunner(
            env=env,
            data_dir=args.data_dir,
            robots=robots,
            max_runner_steps=args.tsteps,
            num_runs=args.num_runs,
            run_name=args.run_name,
            overwrite=True,
            skip_display=args.skip_display,
            llm_output_mode=args.output_mode,  # "action_only" or "action_and_path"
            llm_comm_mode=args.comm_mode,  # "chat" or "plan"
            llm_num_replans=args.num_replans,
            policy_kwargs=dict(
                control_freq=args.control_freq,
                use_weld=args.use_weld,
                skip_direct_path=0,
                skip_smooth_path=0,
                check_relative_pose=args.rel_pose,
            ),
            direct_waypoints=args.direct_waypoints,
            max_failed_waypoints=args.max_failed_waypoints,
            debug_mode=args.debug_mode,
            split_parsed_plans=args.split_parsed_plans,
            use_history=(not args.no_history),
            use_feedback=(not args.no_feedback),
            temperature=args.temperature,
            llm_source=args.llm_source,
            tree_load=args.tree_load,
            skip_check_memory=args.skip_check_memory,
        )
        runner.run(args)
    elif args.task in ['blocksworld']:
        env = BlocksWorldTask(initial_coef=0.7, target_coef=0.3)
        env.set_optimal_steps(args.optimal_steps)
        if args.no_feedback:
            assert args.num_replans == 1, "no feedback mode requires num_replans=1 but longer -tsteps"

        runner = TextLLMRunner(
            env=env,
            data_dir=args.data_dir,
            max_runner_steps=args.tsteps,
            num_runs=args.num_runs,
            run_name=args.run_name,
            overwrite=True,
            skip_display=args.skip_display,
            llm_num_replans=args.num_replans,
            debug_mode=args.debug_mode,
            use_history=(not args.no_history),
            use_feedback=(not args.no_feedback),
            temperature=args.temperature,
            llm_source=args.llm_source,
            tree_load=args.tree_load,
            skip_check_memory=args.skip_check_memory,
        )
        runner.run(args)
    else:
        raise ValueError(f"Task {args.task} not supported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # special parameters in LAQ
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--task", type=str, default="sandwich_dialog")  # "sandwich_dialog", "sort_dialog", "sandwich_central", "sort_central", "blocksworld"
    parser.add_argument("--start_run_id", "-sri", type=int, default=0)
    parser.add_argument("--start_case_id", "-sci", type=int, default=0)
    parser.add_argument("--tree_load", "-tl", action="store_true")
    parser.add_argument("--llm_source", "-llm", type=str, default="gpt-4-0125-preview")
    # optimal steps for different domains
    # [2, 4, 6, 8, 10, 12] for Blocksworld
    # [1, 2, 3, 4, 5, 6] for Sort
    # [6, 8, 10] for Sandwich
    parser.add_argument("--optimal_steps", "-ostep", type=int, default=2)
    parser.add_argument("--temperature", "-temp", type=float, default=0)
    parser.add_argument("--tsteps", "-t", type=int, default=20)     # 20 for Blocksworld, 16 for Sandwich, 8 for Sort

    # parameters in RoCo codes, some is not used in LAQ. We keep them for compatibility.
    parser.add_argument("--run_name", "-rn", type=str, default="test")
    parser.add_argument("--cycle_id", "-ci", type=int, default=0)
    parser.add_argument("--start_id", "-sid", type=int, default=-1)
    parser.add_argument("--num_runs", '-nruns', type=int, default=1)
    parser.add_argument("--output_mode", type=str, default="action_only", choices=["action_only", "action_and_path"])
    parser.add_argument("--comm_mode", type=str, default="dialog", choices=["chat", "plan", "dialog"])
    parser.add_argument("--control_freq", "-cf", type=int, default=15)
    parser.add_argument("--skip_display", "-sd", action="store_true")
    parser.add_argument("--direct_waypoints", "-dw", type=int, default=5)
    parser.add_argument("--num_replans", "-nr", type=int, default=5)
    parser.add_argument("--cont", "-c", action="store_true")
    parser.add_argument("--load_run_name", "-lr", type=str, default="sort_task")
    parser.add_argument("--load_run_id", "-ld", type=int, default=0)
    parser.add_argument("--max_failed_waypoints", "-max", type=int, default=1)
    parser.add_argument("--debug_mode", "-i", action="store_true")
    parser.add_argument("--use_weld", "-w", type=int, default=1)
    parser.add_argument("--rel_pose", "-rp", action="store_true")
    parser.add_argument("--split_parsed_plans", "-sp", action="store_true")
    parser.add_argument("--no_history", "-nh", action="store_true")
    parser.add_argument("--no_feedback", "-nf", action="store_true")
    parser.add_argument("--pre_imagine", "-pi", action="store_true")
    parser.add_argument("--skip_check_memory", "-sca", action="store_true")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args)
