import re
import copy
import json
import numpy as np
from pydantic import dataclasses, validator
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import dm_control
from dm_control.utils.transformations import mat_to_quat
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_SUCTION_CONSTANTS, HUMANOID_CONSTANTS, HUMANOID_LEFT_BODY_NAMES

SANDWICH_TASK_OBJECTS = [
    "table",
    "cutting_board",
    "bread_slice1",
    "bread_slice2",
    "bacon",
    "cheese",
    "tomato",
    "cucumber",
    "ham",
    "beef_patty",
]
SANDWICH_RECIPES = {
    "bacon": ["bread_slice1", "bacon", "cheese", "tomato", "bread_slice2"],
    "vegetarian": ["bread_slice1", "cheese", "tomato", "cucumber", "bread_slice2"],
    "beef_patty": ["bread_slice1", "beef_patty", "cheese", "tomato", "bread_slice2"],
    "ham": ["bread_slice1", "ham", "cheese", "tomato", "cucumber", "bread_slice2"],
}
bacon_recipe = ", ".join(SANDWICH_RECIPES["bacon"])
available_obj2 = ["cutting_board", "bread_slice1", "bread_slice2", "bacon", "cheese", "tomato", "cucumber", "ham", "beef_patty"]
available_obj2_str = ", ".join(available_obj2)

SANDWICH_DIALOG_ACTION_SPACE = """[Action Options]
1) PICK <obj>, Only PICK if gripper is empty. PICK only the correct next item according to the recipe. PICK the food item only if the food item's state is not 'atop <obj>'
2) PUT <obj1> <obj2>. <obj1> can be one of the foods. <obj2> can be food or cutting_board
3) WAIT, do nothing
Only one robot can PUT each round. You must PICK up an item before PUT
[Action Output Instruction]
Must first output 'EXECUTE\n', then give exactly one action per robot, put each on a new line
Example#1: 'EXECUTE\nNAME Chad ACTION PUT bread_slice1 cutting_board\nNAME Dave ACTION PICK tomato\n'
Example#2: 'EXECUTE\nNAME Chad ACTION WAIT\nNAME Dave ACTION PUT cheese tomato\n'
1. The line of the actions should be less than or equal to two. The first agent should be Chad and the second agent should be Dave
2. The number of WAIT should be less than or equal to one. The number of PUT should be less than or equal to one
3. PUT <obj2> only if the state of <obj2> is 'atop [food item or cutting_board]' and no other food item is 'atop <obj2>'"""

SANDWICH_ACTION_CONSTRAINTS = f"""[Detailed Constraints for Action]
Check the following constraints and Fill in blanks in '[]'. Check these constraints one by one: 1, 2...
* Get [Extended Recipe Order]: [cutting_board, bread_slice1, ...] (cutting_board + [Recipe Order])
* Repeat <B-Item>=[Chad: [], Dave: []]
1. [Line Number Check] The line of the actions should equal to two
2. [Agent Order Check] The first agent should be Chad and the second agent should be Dave
3. [WAIT Number Check] The number of WAIT should be less than or equal to one
4. [PUT Number Check] The number of PUT should be less than or equal to one
5. [PICK Side Check] PICK the food item on the side of the robot
6. [PICK Robot State Check] PICK the food item only if the gripper is empty
7. [PICK Food State Check] PICK the food item only if the food item's state is not 'atop <obj>'
8. [PUT Availability Check 1] If <obj2> is not cutting_board, repeat <obj2>: [], state of <obj2>: <State-2>=[]. Get food item before <obj2> in the [Extended Recipe Order]: <X-item>=[]. 'atop <X-item>' is: <State-X>=[]. <State-2> should equal to <State-X>: []
9. [PUT Availability Check 2] Get <item>s in [State] whose state matches '<item>: atop <obj2>': pool=[]. If no other food item is 'atop <obj2>' (meaning pool is empty), this constraint is valid directly: []
10. [PUT Order Check 1] Repeat <obj1>: [], <obj1> should equal to <B-Item>=[]: []
11. [PUT Order Check 2] Repeat <obj1>: [], get food item before <obj1> in [Extended Recipe Order]: <Y-item>=[], <obj2>: []. <obj2> should equal to <Y-Item>=[]: [].
12. [PICK Order Check] For PICK action, robot should PICK the first food item in its [Reachable Items]: []
When checking, you should repeat the prompt and fill in blanks and give the line conclusion  (yes/no/not applicable). For example:
10. [PUT Order Check 1] Repeat <obj1>: [bread_slice2], <obj1> should equal to <B-Item>=[bread_slice2]: [yes]
11. [PUT Order Check 2] Repeat <obj1>: [bread_slice2], get food item before <obj1> in [Extended Recipe Order]: <Y-item>=[cucumber], <obj2>: [cucumber]. <obj2> should equal to <Y-Item>=[cucumber]: [yes]. Line conclusion: [yes]
"""

SANDWICH_STATE_SPACE = """[State Space Definition]
Define the state of the multi-agent system, which is composed of two categories: [Food States] and [Robot States]
1. [Food States]: Describe the status of the following eight food items: bread_slice1, bread_slice2, bacon, cheese, tomato, cucumber, ham, beef_patty
  + The food items must be listed in this order: bread_slice1, bread_slice2, bacon, cheese, tomato, cucumber, ham, beef_patty
2. [Robot States]: Specify the current status of each robot's gripper

[State Template]
[State]
[Food States]
bread_slice1: <food state>
bread_slice2: <food state>
bacon: <food state>
cheese: <food state>
tomato: <food state>
cucumber: <food state>
ham: <food state>
beef_patty: <food state>
[Robot States]
1. Chad's gripper is <gripper state>
2. Dave's gripper is <gripper state>

[Detailed Constraints for State]
The following paragraphs are the detailed constraints for State. There are three main <check items>: [Title Check], [Food State Check], and [Robot State Check]
1. [Title Check] A header titled [State]
2. [Food State Check] A header titled [Food States]. Then, list the Food States
  + Each line represents the state of one food item
  + The food items must be listed in this order: [bread_slice1, bread_slice2, bacon, cheese, tomato, cucumber, ham, beef_patty]
  + The <food state> specify the location of each food item. The <food item> can only be one of these states:
    - on right side
    - on left side
    - atop [another food item or cutting_board]
    - gripped by [robot name]
3. [Robot State Check] A header titled [Robot States]. Then, list the Robot States
  + Each line represents the state of one robot's gripper
  + Each robot's gripper can be in one of two states: 'empty', or 'holding [food item]'

Note: The system includes eight food items and one cutting board by default"""

SANDWICH_PREDICTING_PROMPT = f"""[Predicting Instruction]
You will be provided with the [State] and the [Action] of the multi-agent system. You should think step by step to output the [Prediction] of the next [State] based on the given [State] and [Action]. The format of the [Prediction] should follow the [Detailed Constraints for State]
Please output your thinking process step-by-step by following theses steps: [Interaction Item Pool], [Action Forward Rule] and [Prediction Conclusion]
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content
1. [Interaction Item Pool]: initialize the pool of the food items that the robots are interacting with. It is a empty list at the beginning: pool={{}}
2. [Action Forward Rule]: the state in the [Prediction] is changed by the [Action]. Follow these steps to predict the [Prediction]:
  + List the action of <agent>=[Chad and Dave]: []
    - For WAIT, the state of <agent> should not change
    - For PICK, repeat action: [], <obj>: [], <agent>: []. State of <agent> should be '<agent>'s gripper is holding <obj>'. State of the <obj> should be 'gripped by <agent>'. Add <obj> to the pool: pool=[]
    - For PUT, repeat action: [], <obj1> :[], <obj2>: []. State of <agent> should be '<agent>'s gripper is empty'. State of the <obj1> should be 'atop <obj2>'. Add <obj1> to the pool: pool=[]
3. [Prediction Conclusion]: conclude the [Prediction] based on the [Action Forward Rule]
  + The format of the [Prediction] should follow the [Detailed Constraints for State]
  + The food items that are not present in the pool should not change their state
Example:
[State] ...
[Action] ...
[Prediction] is：
[State] ..."""

SANDWICH_PREDICTING_CONSTRAINTS = f"""[Detailed Constraints for Prediction]
Check the following constraints and Fill in blanks in '[]'. Check these constraints one by one: 1, 2... Only get your conclusion according to the results of these checks! 
Initialize a empty <pool>={{}} as Interaction Items
* Get [Interaction Items Pool]: first initialize an empty <pool>={{}}. For PICK actions, repeat <obj>: [], add <obj> to <pool>=[]. For PUT action, repeat <obj1>: [], add <obj1> to <pool>=[].
1. [Header Title Check] A header titled [State] -> A header titled [Food States] -> ... -> A header titled [Robot States] -> ...: []
2. [Food Order Check] Food items must be listed in this order: [bread_slice1, bread_slice2, bacon, cheese, tomato, cucumber, ham, beef_patty]: []
3. [Food Format Check] The <food item> should be one of: on right side, on left side, atop <another food item>, atop cutting_board, and gripped by <robot>: []
4. [Robot Format Check] Each line represents the state of one robot's gripper. Each robot's gripper should be one of: 'empty' and 'holding <food item>': []
5. [PICK Rule Check 1] For PICK, repeat <agent>: [], <obj>: [], state of <agent>: [], it should equal to '<agent>'s gripper is holding <obj>': []
6. [PICK Rule Check 2] For PICK, repeat <agent>: [], <obj>: [], state of <obj> in prediction: [], it should equal to 'gripped by <agent>': []
7. [PUT Rule Check 1] For PUT, repeat <agent>: [], repeat state of <agent>: [], it should equal to '<agent>'s gripper is empty': []
8. [PUT Rule Check 2] For PUT, repeat <obj1>=[], <obj2>=[], <State-1>=[atop <obj2>], State of <obj1> in prediction: <State-2>=[]. <State-2> should equal to <State-1>: []
9. [Other State Check] Repeat items not in <pool>=[]: []. State of these items should not change: []
* When checking, you should repeat the prompt and fill in blanks, replace <obj> and <agent> with their true values, and give the line conclusion (yes/no/not applicable). For example:
8. [PUT Rule Check 2] For PUT, repeat <obj1>=[cheese], <obj2>=[cucumber], <State-1>=[atop cucumber]. State of <obj1> in prediction: <State-2>=[atop cucumber]. <State-2> should equal to <State-1>: [yes]. Line conclusion: [yes]
9. [Other State Check] Repeat items not in <pool>=[cucumber]: [bread_slice1, bread_slice2, bacon, cheese, tomato, ham, beef_patty]. State of these items should not change: [yes]. Line conclusion: [yes]
* Constraint 5 & 6 MAY have two lines. If so, you should output results separately. For example: 
5.1 [PICK Rule Check 1] For PICK, repeat <agent>: [Chad], <obj>: [cucumber], state of <agent>: [Chad's gripper is holding cucumber], it should equal to '<agent>'s gripper is holding <obj>': [yes]. Line conclusion: [yes]
5.2 [PICK Rule Check 1] For PICK, repeat <agent>: [Dave], ... 
"""


SANDWICH_CHAT_PROMPT = """The robots discuss before taking actions. Carefully consider environment feedback and others' responses, and coordinate to strictly follow the sandwich recipe and avoid collision.
They talk in order [Chad],[Dave],[Chad],..., after reaching agreement, they output a plan with **exactly** one ACTION per robot, and stop talking. Their chat and final plan are: """

SANDWICH_PLAN_PROMPT = """
Plan one ACTION for each robot at every round. The robot ACTIONs must strictly follow the sandwich recipe and avoid collision.
"""


class MakeSandwichTask(MujocoSimEnv):
    def __init__(
            self,
            filepath: str = "rocobench/envs/task_sandwich.xml",
            one_obj_each: bool = False,
            **kwargs,
    ):
        self.robot_names = ["ur5e_suction", "humanoid"]
        self.agent_names = ["Chad", "Dave"]
        self.robot_name_map = {
            "ur5e_suction": "Chad",
            "humanoid": "Dave",
        }
        self.robot_name_map_inv = {
            "Chad": "ur5e_suction",
            "Dave": "humanoid",
        }
        self.robots = dict()
        self.food_items = SANDWICH_TASK_OBJECTS[2:]  # exclude cutting board and table

        self.cases_information = []
        for sandwich_type in ['bacon', 'vegetarian', 'beef_patty', 'ham']:
            with open('rocobench/envs/task_sandwich_cases/{}.json'.format(sandwich_type), 'r') as f:
                temp_cases = json.load(f)
                for case in temp_cases:
                    case['sandwich_type'] = sandwich_type
                    self.cases_information.append(case)

        self.full_cases = {}
        for optimal_step in range(20):
            temp_list = []
            for case in self.cases_information:
                if case['optimal_steps'] == optimal_step:
                    temp_list.append([case['goal_state'], case['sandwich_type']])

            if len(temp_list) > 0:
                self.full_cases[optimal_step] = temp_list

        self.cases = None
        self.set_optimal_steps(6)

        super(MakeSandwichTask, self).__init__(
            filepath=filepath,
            task_objects=SANDWICH_TASK_OBJECTS,
            agent_configs=dict(
                ur5e_suction=UR5E_SUCTION_CONSTANTS,
                humanoid=HUMANOID_CONSTANTS,
            ),
            skip_reset=True,
            **kwargs
        )
        self.cutting_board_pos = self.physics.data.body("cutting_board").xpos.copy()
        self.name = 'sandwich_dialog'

        all_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                all_panels.append(
                    (geom.name, geom.pos, geom.size)
                )
        assert len(all_panels) >= len(self.food_items), "Not enough panel positions to sample from"
        self.all_panels = all_panels
        self.left_panels = [p for p in all_panels if p[1][0] < 0]
        self.right_panels = [p for p in all_panels if p[1][0] > 0]
        self.reset(keyframe_id=0, home_pos=None, reload=False)  # special case for this task

        suction_config = UR5E_SUCTION_CONSTANTS.copy()
        self.robots[
            self.robot_name_map["ur5e_suction"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **suction_config,
        )
        humanoid_config = HUMANOID_CONSTANTS.copy()
        self.robots[
            self.robot_name_map["humanoid"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **humanoid_config,
        )

        self.align_threshold = 0.15
        self.recipe_order = SANDWICH_RECIPES["bacon"]
        self.reachable_items = None

    @property
    def use_prepick(self):
        return True

    @property
    def use_preplace(self):
        return True

    def set_optimal_steps(self, optimal_steps):
        self.optimal_steps = optimal_steps
        self.cases = self.full_cases[optimal_steps]
        self.case_number = len(self.cases)

    def process_state(self, state):
        state_lines = state.split("\n")
        state_start, robot_state_start = len(state_lines) - 14, len(state_lines) - 14
        state_start, robot_state_start = max(0, state_start), max(0, robot_state_start)
        for line in state_lines:
            if '[State]' in line:
                state_start = state_lines.index(line)
            if '[Robot States]' in line:
                robot_state_start = state_lines.index(line)

        state_start = min(state_start, robot_state_start)
        state_lines = state_lines[state_start:state_start + 13]

        food_state_dict = {}
        for line in state_lines:
            for food in self.food_items:
                if food + ': ' in line:
                    # line is like 'bacon: on right side', but sometimes there are more words after the food name
                    _, state_str = re.match(r'(.+): (.+)', line).groups()
                    food_state_dict[food] = '{}: {}'.format(food, state_str)
            for agent in self.agent_names:
                if agent + "'s" in line:
                    # line is like 'Chad's gripper is empty'
                    _, state_str = re.match(r'(.+) is (.+)', line).groups()
                    if agent == 'Chad':
                        food_state_dict[agent] = "1. Chad's gripper is {}".format(state_str)
                    elif agent == 'Dave':
                        food_state_dict[agent] = "2. Dave's gripper is {}".format(state_str)
                    else:
                        raise ValueError('Unknown agent name')

        processed_state = '[State]\n'
        processed_state += '[Food States]\n'
        for food in self.food_items:
            if food in food_state_dict.keys():
                processed_state += food_state_dict[food] + '\n'

        processed_state += '[Robot States]\n'
        for agent in self.agent_names:
            if agent in food_state_dict.keys():
                processed_state += food_state_dict[agent] + '\n'

        return processed_state

    def get_state_template(self):
        state_template = f"""[State Template]
[State]
[Food States]
bread_slice1: <food state>
bread_slice2: <food state>
bacon: <food state>
cheese: <food state>
tomato: <food state>
cucumber: <food state>
ham: <food state>
beef_patty: <food state>
[Robot States]
1. Chad's gripper is <gripper state>
2. Dave's gripper is <gripper state>
"""
        return state_template

    def get_check_tail(self):
        predicting_tail = f"""[Start]
[Interaction Items Pool]: first initialize ...
1. [Header Title Check] ...
[Conclusion] ..."""
        action_tail = f"""[Start]
[Extended Recipe Order]: ..., 
<B-Item>=[Chad: ...; Dave: ...]  
1. [Line Number Check] ..."""
        return {'predicting': predicting_tail, 'action': action_tail}

    def get_policy_tail(self):
        prompt = f"""
Since the next food item should be PUT on the sandwich is ..., and ..."""
        return prompt

    def process_action(self, action):
        action_lines = action.split("\n")
        agent_action_dict = {}
        for line in action_lines:
            for agent in self.agent_names:
                if f'NAME {agent}' in line:
                    agent_action_dict[agent] = line

        processed_action = 'EXECUTE\n'
        for agent in self.agent_names:
            if agent in agent_action_dict.keys():
                processed_action += agent_action_dict[agent] + '\n'

        if 'ERROR' in action or 'error' in action or 'Error' in action:
            processed_action = 'ERROR\n'

        return processed_action

    def get_target_pos(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if target_name in self.food_items + ["cutting_board"]:
            sname = target_name
            ret = self.physics.data.site(sname).xpos.copy()
        elif target_name == "table":
            if agent_name == "Chad" or robot_name == "ur5e_suction":
                panels = self.right_panels
            else:
                panels = self.left_panels
            # pick the panel that's farthest from all the objects on this side of table
            empty_pos = panels[0][1]
            for p in panels:
                dist_to_objs = [
                    np.linalg.norm(
                        self.physics.data.site(obj).xpos[:2] - p[1][:2]
                    ) for obj in self.food_items
                ]
                if all([d > 0.1 for d in dist_to_objs]):
                    empty_pos = p[1]
                    break
            ret = empty_pos
        return ret

    def get_target_quat(self, agent_name, target_name) -> Optional[np.ndarray]:
        ret = None
        if target_name in self.food_items + ["cutting_board"]:
            try:
                xmat = self.physics.data.site(target_name).xmat.copy()
                ret = mat_to_quat(xmat.reshape(3, 3))
            except KeyError:
                pass
        if target_name == "table":
            ret = np.array([1, 0, 0, 0])
        return ret

    def get_target_prompt(self, obs: EnvState):
        # get the prompt of the final state
        # task specific!
        state_desp = self.get_state_prompt(obs)
        state_desp_dict = {}
        for food_item in self.food_items:
            if food_item + ': on right side' in state_desp:
                state_desp_dict[food_item] = food_item + ': on right side\n'
            elif food_item + ': on left side' in state_desp:
                state_desp_dict[food_item] = food_item + ': on left side\n'
            else:
                raise ValueError('Food item not on table')

        target_state_desp = f"""[State]
[Food States]
"""
        for food_item in self.food_items:
            if food_item not in self.recipe_order:
                target_state_desp += state_desp_dict[food_item]
            else:
                if food_item == self.recipe_order[0]:
                    target_state_desp += food_item + ': atop cutting_board\n'
                else:
                    target_state_desp += food_item + ': atop {}\n'.format(self.recipe_order[self.recipe_order.index(food_item) - 1])

        target_state_desp += f"""[Robot States]
1. Chad's gripper is empty
2. Dave's gripper is empty
"""
        self.target_state_desp = target_state_desp

    def get_graspable_objects(self):
        return dict(
            Chad=self.food_items,
            Dave=self.food_items,
        )

    def get_grasp_site(self, obj_name: str = "cheese") -> Optional[str]:
        if obj_name in SANDWICH_TASK_OBJECTS:
            return obj_name
        else:
            return None

    def get_reward_done(self, obs):
        # task specific!
        rew = 0
        done = False
        board_contacts = obs.objects['cutting_board'].contacts
        if len(board_contacts) == 0:
            return 0, False
        elif 'bread_slice1' not in board_contacts:
            return 0, True

        for i, item in enumerate(self.recipe_order):
            item_contacts = obs.objects[item].contacts
            if len(item_contacts) == 0:
                return 0, False
            if i == len(self.recipe_order) - 1:
                return 1, True
            next_item = self.recipe_order[i + 1]
            if next_item not in item_contacts:
                return 0, False

        return rew, done

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "humanoid" or robot_name == self.robot_name_map["humanoid"]:
            return dict(x=(-1.4, 0.1), y=(0.3, 1.5), z=(0.16, 1))
        elif robot_name == "ur5e_suction" or robot_name == self.robot_name_map["ur5e_suction"]:
            return dict(x=(-0.1, 1.3), y=(-0.2, 0.7), z=(0.16, 1))
        else:
            raise NotImplementedError

    def sample_initial_scene(self, case_id: int = 0):
        # sample locations of the pan 
        sampled_panels = []
        n_left_items = len(self.food_items) // 2
        left_idxs = self.random_state.choice(
            len(self.left_panels), size=n_left_items, replace=False
        )
        sampled_panels.extend(
            [self.left_panels[i] for i in left_idxs]
        )

        n_right_items = len(self.food_items) - n_left_items
        right_idxs = self.random_state.choice(
            len(self.right_panels), size=n_right_items, replace=False
        )
        sampled_panels.extend(
            [self.right_panels[i] for i in right_idxs]
        )

        left_items, right_items = [], []
        for food, (_, pos, size) in zip(self.food_items, sampled_panels):
            # new_quat = Quaternion(
            #     axis=[0,0,1], angle=self.random_state.uniform(low=0, high=np.pi*2)
            #     ) 

            # new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z]) 
            new_quat = None  # TODO: fix the picking quat before enable sampling
            new_pos = self.random_state.uniform(
                low=pos - size / 2, high=pos + size / 2
            )
            new_pos[2] = 0.2
            self.reset_body_pose(
                body_name=food,
                pos=new_pos,
                quat=new_quat,
            )
            self.reset_qpos(
                jnt_name=f"{food}_joint",
                pos=new_pos,
                quat=new_quat,
            )
            if pos[0] < 0:
                left_items.append(food)
            else:
                right_items.append(food)

        self.physics.forward()
        self.physics.step(100)  # let the food drop
        # sampling a random recipe!!
        # recipe_idx = self.random_state.choice(
        #     len(SANDWICH_RECIPES), size=1, replace=False)[0]

        # NOTE: set the recipe to be the same for all episodes temporarily
        # recipe_idx = 0
        # recipe = list(SANDWICH_RECIPES.keys())[recipe_idx]
        # recipe_order = SANDWICH_RECIPES[recipe]
        # # randomly shuffle the order
        # food_items = recipe_order[1:-1].copy()
        # self.random_state.shuffle(food_items)
        # recipe_order[1:-1] = food_items

        case_goal_state, sandwich_type = self.cases[case_id]
        temp_item, recipe, end_flag = 0, [], False
        while not end_flag:
            replace_flag = False
            for i in range(1, 9):
                if case_goal_state[str(i)] == temp_item:
                    recipe.append(i)
                    temp_item = i
                    replace_flag = True
                    break

            if not replace_flag:
                end_flag = True

        recipe_order = []
        for item in recipe:
            recipe_order.append(self.food_items[item - 1])

        reachable_items = {
            'Chad': [],
            'Dave': []
        }
        # Chad for right side, Dave for left side
        for item in recipe_order:
            if item in right_items:
                reachable_items['Chad'].append(item)

        for item in recipe_order:
            if item in left_items:
                reachable_items['Dave'].append(item)

        self.reachable_items = reachable_items
        self.recipe_order = recipe_order
        self.recipe_name = f"{sandwich_type}_sandwich"
        self.recipe = sandwich_type

    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        table_id = self.physics.model.body("table").id
        board_id = self.physics.model.body("cutting_board").id
        food_ids = [self.physics.model.body(food).id for food in self.food_items]

        ret = []
        for food_id in food_ids:
            ret.append((food_id, table_id))
            ret.append((food_id, board_id))
            for food_id2 in food_ids:
                if food_id != food_id2:
                    ret.append((food_id, food_id2))

        for link_id in self.robots["Chad"].all_link_body_ids + self.robots["Dave"].all_link_body_ids:
            for food_id in food_ids + [board_id]:
                ret.append((link_id, food_id))

        # humanoid left arm is allowed to touch the table
        for link_name in HUMANOID_LEFT_BODY_NAMES:
            link_id = self.physics.model.body(link_name).id
            ret.append((link_id, table_id))
        # special case for suction gripper sometimes it can touch the table
        ret.extend(
            [
                (self.physics.model.body("ur5e_suction").id, table_id),
                (self.physics.model.body("rpalm").id, table_id),
            ]

        )
        return ret

    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs

    def describe_food_state(self, obs, item_name: str = "bacon") -> str:
        food_state = obs.objects[item_name]
        food_desp = f"{item_name}: "
        contacts = food_state.contacts
        if 'table' in contacts and 'cutting_board' not in contacts:
            side = 'left' if food_state.xpos[0] < 0 else 'right'
            food_desp += f"on {side} side"
        elif 'cutting_board' in contacts:
            xpos = obs.objects['cutting_board'].xpos
            if xpos[2] < food_state.xpos[2]:
                food_desp += f"atop cutting_board"
        elif any([f in contacts for f in self.food_items]):
            for f in self.food_items:
                xpos = obs.objects[f].xpos
                if f != item_name and f in contacts and xpos[2] < food_state.xpos[2]:
                    food_desp += f"atop {f}"
        else:
            if 'ur5e_suction' in contacts:
                food_desp += f"gripped by Chad"
            elif 'humanoid' in contacts:
                food_desp += f"gripped by Dave"
            else:
                raise ValueError(f"Unknown contact {contacts}")

        return food_desp

    def describe_robot_state(self, obs, robot_name: str = "humanoid") -> str:
        robot_state = getattr(obs, robot_name)
        # x, y, z = robot_state.xpos.tolist()
        # robot_desp += f"{agent_name}'s gripper is at ({x:.2f}, {y:.2f}, {z:.2f}),\n"
        contacts = robot_state.contacts
        agent_name = self.robot_name_map[robot_name]
        if len(contacts) == 0:
            robot_desp = f"{agent_name}'s gripper is empty"
        else:
            obj = ",".join([c for c in contacts])
            robot_desp = f"{agent_name}'s gripper is holding {obj}"
        return robot_desp

    def get_agent_prompt(self, state_desp, agent_name: str = "Chad") -> str:
        assert agent_name in self.agent_names, f"Agent {agent_name} is not in the scene"

        other_robot = [r for r in self.agent_names if r != agent_name][0]
        table_side = "left" if agent_name == "Dave" else "right"
        other_side = "right" if agent_name == "Dave" else "left"

        recipe_str = ", ".join(self.recipe_order)

        state_desp_lines = state_desp.split("\n")
        food_states_dict = {}
        agent_states_dict = {}
        for line in state_desp_lines:
            for food in self.food_items:
                if food + ': ' in line:
                    food_states_dict[food] = line
            for temp_agent in self.agent_names:
                if temp_agent + "'s" in line:
                    agent_states_dict[temp_agent] = line

        # each agent can only see items on their own side of the table or on the cutting board
        food_states, table_states = [], []
        for food in self.food_items:
            desp = food_states_dict[food]
            if other_side not in desp and 'atop' not in desp:
                food_states.append(desp.replace(table_side, "your"))
            if 'atop' in desp:
                table_states.append(desp)

        food_states = "\n".join(food_states)

        if len(table_states) == 0:
            table_states = "There are no food items on the cutting board"
        else:
            table_states = str(table_states)

        current_reachable_items = copy.deepcopy(self.reachable_items)
        state_lines = state_desp.split("\n")
        items_on_cutting_board = []
        for line in state_lines:
            if 'atop' in line:
                food_item = line.split(':')[0]
                items_on_cutting_board.append(food_item)
                if food_item in current_reachable_items['Chad']:
                    current_reachable_items['Chad'].remove(food_item)
                if food_item in current_reachable_items['Dave']:
                    current_reachable_items['Dave'].remove(food_item)

        agent_state = agent_states_dict[agent_name]
        agent_state = agent_state.replace(f"{agent_name}'s", "Your")[3:]
        agent_prompt = f"""
You are a robot {agent_name}, collaborating with {other_robot} to make a [{self.recipe_name}].
[Recipe Order] Food items must be stacked following this order: {recipe_str}, where bread_slice1 must be PUT atop cutting_board. 
You must stay on {table_side} side of the table! This means you can only PICK food from {table_side} side, and {other_robot} can only PICK from the other side.
Only one robot can PUT at a time, so you must coordinate with {other_robot}.

At the current round:
You can see these food items are on your reachable side, and you can only interact with them:
{food_states}
{agent_state}
The table state is:
{table_states}
The [Reachable Items] in [Recipe] of each robot are:
{'Chad: ' + ', '.join(current_reachable_items['Chad'])}
{'Dave: ' + ', '.join(current_reachable_items['Dave'])}
The first item in [Reachable Items] is: <B-Item>=[Chad: {
        current_reachable_items['Chad'][0] if len(current_reachable_items['Chad']) > 0 else 'None'
        }; Dave: {
        current_reachable_items['Dave'][0] if len(current_reachable_items['Dave']) > 0 else 'None'
        }]
Think step-by-step about the task and {other_robot}'s response. Carefully check and correct them if they made a mistake. 
Improve your plans if given [Environment Feedback].
Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan. If you agree with someone, you should tell them until everyone agrees on the plan.
Note: Propose exactly one action for [yourself] at the **current** round, select from [Action Options].
**Note**: End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If **everyone** has made proposals and **got approved**, output the final plan, must strictly follow [Action Output Instruction]!
If everyone has made proposals and got approved, output the final plan with 'EXECUTE\nNAME Chad ACTION ...' and 'NAME Dave ACTION ...'.
An agent cannot make final decisions without getting approval from the other agent.
"""
        return agent_prompt

    def if_fail(self, state_desp: str):
        if 'bread_slice2: atop' in state_desp and state_desp != self.target_state_desp:
            return True

        return False

    def describe_obs(self, obs: EnvState):
        object_desp = "[Scene description]\n"
        for food in self.food_items:
            object_desp += self.describe_food_state(obs, food) + "\n"

        robot_desp = "The robots:\n"
        for robot_name, agent_name in self.robot_name_map.items():
            robot_desp += self.describe_robot_state(obs, robot_name) + "\n"

        full_desp = object_desp + robot_desp
        return full_desp

    def get_task_feedback(self, obs, llm_plan, pose_dict):
        task_feedback = ""
        error_info = {
            'dynamics_error': False,
            'task_error': False
        }

        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' in action_str and 'PUT' in action_str:
                task_feedback += f"{agent_name}'s can't PICK and PUT at same time.\n"
            elif 'PUT' in action_str:
                objects = action_str.split('PUT')[1].strip().split(' ')
                if len(objects) == 2:
                    obj1 = objects[0]
                    obj2 = objects[1]
                    if obj1 not in self.recipe_order:
                        task_feedback += f"{obj1} is not in the recipe\n"
                    elif obj2 == "table":
                        continue
                    else:
                        idx1 = self.recipe_order.index(obj1)
                        if idx1 == 0 and obj2 != 'cutting_board':
                            task_feedback += f"recipe says {obj1} must be put atop cutting_board\n"
                            error_info['task_error'] = True
                        elif idx1 > 0:
                            if obj2 not in self.recipe_order:
                                task_feedback += f"{obj1} is not allowed to be put on {obj2}"
                            else:
                                idx2 = self.recipe_order.index(obj2)
                                if idx2 != idx1 - 1:
                                    task_feedback += f"recipe says {obj1} must be put on {self.recipe_order[idx1 - 1]}\n"
                                    error_info['task_error'] = True
                                else:
                                    obj2_xpos = obs.objects[obj2].xpos
                                    if np.linalg.norm(obj2_xpos[:2] - self.cutting_board_pos[:2]) > 0.4:
                                        task_feedback += f"{obj2} is not atop cutting_board\n"
            elif 'PICK' in action_str:
                obj = action_str.split('PICK')[1].strip()
                if obj in self.food_items:
                    contacts = obs.objects[obj].contacts
                    if 'cutting_board' in contacts or any([f in contacts for f in self.food_items]):
                        task_feedback += f"{agent_name} cannot PICK {obj}, it's already stacked\n"
                # Chad for right side, Dave for left side
                if agent_name == 'Chad' and obj in ['bread_slice1', 'bread_slice2', 'bacon', 'cheese']:
                    task_feedback += f"{agent_name} cannot PICK {obj}, it's on left side\n"
                elif agent_name == 'Dave' and obj in ['tomato', 'cucumber', 'ham', 'beef_patty']:
                    task_feedback += f"{agent_name} cannot PICK {obj}, it's on right side\n"

        if all(['PUT' in action_str for action_str in llm_plan.action_strs.values()]):
            task_feedback += "only one robot can PUT at a time\n"

        if len(task_feedback) > 0 and not error_info['task_error']:
            error_info['dynamics_error'] = True

        return task_feedback, error_info

    def describe_robot_capability(self):
        return ""

    def describe_task_context(self):
        recipe_str = ", ".join(self.recipe_order)
        context = f"""2 robots, Chad and Dave, together make a [{self.recipe_name}].
Food items must be stacked following this order: {recipe_str}, where bread_slice1 must be PUT atop cutting_board. 
Chad can only reach right side of the table, and Dave can only reach left side of the table.
Both robots can PICK food items, or PUT an item atop something; only one robot can PUT at a time. 
At each round, given [Scene description] and [Environment feedback], use it to reason about the task and improve plans.
"""
        return context

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix!  
        contacts["ur5e_suction"] = [c for c in contacts["ur5e_suction"] if c in self.food_items]
        contacts["humanoid"] = [c for c in contacts["humanoid"] if c in self.food_items]
        return contacts

    def get_action_constrains(self, state_desp: str = ""):
        current_reachable_items = copy.deepcopy(self.reachable_items)
        state_lines = state_desp.split("\n")
        items_on_cutting_board = []
        for line in state_lines:
            if 'atop' in line:
                food_item = line.split(':')[0]
                items_on_cutting_board.append(food_item)
                if food_item in current_reachable_items['Chad']:
                    current_reachable_items['Chad'].remove(food_item)
                if food_item in current_reachable_items['Dave']:
                    current_reachable_items['Dave'].remove(food_item)

        prompt = f"""[Action Options]
1) PICK <obj>: Pick one food <item>. Only PICK if gripper is empty. PICK only the correct next item according to the recipe
2) PUT <obj1> <obj2>: PUT <obj1> on the top of <obj2>. <obj1> can be one of the foods. <obj2> can be food or cutting_board
3) WAIT, do nothing
Only one robot can PUT each round. You must PICK up an item before PUT
[Action Output Instruction]
Must first output 'EXECUTE\n', then give exactly one action per robot, put each on a new line
Dave can only pick up the food item on the left side of the table. Chad can only pick up the food item on the right side of the table

"""
        prompt += 'The [Reachable Items] in [Recipe] of each robot are: \n'
        prompt += 'Chad: ' + ', '.join(current_reachable_items['Chad']) + '\n'
        prompt += 'Dave: ' + ', '.join(current_reachable_items['Dave']) + '\n'
        prompt += 'The first item in [Reachable Items] is: <B-Item>=[Chad: {}; Dave: {}]'.format(
            current_reachable_items['Chad'][0] if len(current_reachable_items['Chad']) > 0 else 'None',
            current_reachable_items['Dave'][0] if len(current_reachable_items['Dave']) > 0 else 'None'
        )
        prompt += 'These items contain those gripped by the robots. Chad can only reach right side, and Dave can only reach left side.\n'
        prompt += SANDWICH_ACTION_CONSTRAINTS

        return prompt

    def get_prediction_constrains(self):
        prompt = SANDWICH_PREDICTING_CONSTRAINTS

        return prompt

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return SANDWICH_CHAT_PROMPT

    def central_plan_prompt(self):
        return SANDWICH_PLAN_PROMPT

    def get_action_prompt(self) -> str:
        return SANDWICH_DIALOG_ACTION_SPACE

    def get_action_space_prompt(self) -> str:
        return SANDWICH_DIALOG_ACTION_SPACE

    def get_state_space_prompt(self) -> str:
        return SANDWICH_STATE_SPACE

    def get_predicting_prompt(self) -> str:
        return SANDWICH_PREDICTING_PROMPT

    def get_observation_prompt(self, obs, agent_name: str = "Chad") -> str:
        assert agent_name in self.agent_names, f"Agent {agent_name} is not in the scene"
        other_robot = [r for r in self.agent_names if r != agent_name][0]
        table_side = "left" if agent_name == "Dave" else "right"
        other_side = "right" if agent_name == "Dave" else "left"

        recipe_str = ", ".join(self.recipe_order)

        # each agent can only see items on their own side of the table or on the cutting board
        food_states = []
        for food in self.food_items:
            desp = self.describe_food_state(obs, food)
            if (other_side not in desp and len(desp) > 0):
                food_states.append(desp.replace(table_side, "your"))
        food_states = "\n".join(food_states)

        robot_name = self.robot_name_map_inv[agent_name]
        agent_state = self.describe_robot_state(obs, robot_name)
        agent_state = agent_state.replace(f"{agent_name}'s", "Your")
        agent_prompt = f"""You are a robot {agent_name}, collaborating with {other_robot} to make a [{self.recipe_name}].
Food items must be stacked following this order: {recipe_str}, where bread_slice1 must be PUT atop cutting_board. 
You must stay on {table_side} side of the table! This means you can only PICK food from {table_side} side, and {other_robot} can only PICK from the other side.
Only one robot can PUT at a time, so you must coordinate with {other_robot}.

At the current round:
You can see these food items are on your reachable side:
{food_states}
{agent_state}
Think step-by-step about the task and {other_robot}'s response. Carefully check and correct them if they made a mistake. 
Improve your plans if given [Environment Feedback].
Respond very concisely but informatively, and do not repeat what others have said. Discuss with others to come up with the best plan.
Propose exactly one action for yourself at the **current** round, select from [Action Options].
End your response by either: 1) output PROCEED, if the plans require further discussion; 2) If everyone has made proposals and got approved, output the final plan, must strictly follow [Action Output Instruction]!
"""
        return agent_prompt

    def get_state_prompt(self, obs):
        # the table side is right for Chad and left for Dave
        # each agent can only see items on their own side of the table or on the cutting board
        food_states = []
        for food in self.food_items:
            desp = self.describe_food_state(obs, food)
            food_states.append(desp)
        food_states = "\n".join(food_states)

        robot_names = self.robot_name_map_inv.values()
        gripper_states = {}
        for robot_name in robot_names:
            agent_state = self.describe_robot_state(obs, robot_name)
            gripper_states[robot_name] = agent_state

        state_prompt = f"""[State]
[Food States]
{food_states}
[Robot States]
1. {gripper_states[self.robot_name_map_inv['Chad']]}
2. {gripper_states[self.robot_name_map_inv['Dave']]}
"""
        return state_prompt

    def get_task_prompt(self):
        recipe_str = ", ".join(self.recipe_order)
        task_prompt = f"""[Task Description]
Task: Cooperative Cooking in a Multi-Agent System

Agents: Chad and Dave
Chad - Can only PICK food items from the right side of the table.
Dave - Can only PICK food items from the left side of the table.

Objective: Collaboratively prepare a meal named "[{self.recipe_name}]". The food items of target state must be assembled in the following sequence: [Recipe Order]=[{recipe_str}]. 
"""
        return task_prompt

    def get_repeat_template(self):
        state_template = '[bread_slice1: [], bread_slice2: [], bacon: [], cheese: [], tomato: [], cucumber: [], ham: [], beef_patty: []; Chad: [], Dave:[]]'
        action_template = '[Chad: [], Dave:[]]'

        return {'state': state_template, 'action': action_template}

    def get_examples(self):
        plan_example = f"""[Plan Example]
Example#1:
[Original State]
[Food States]
bread_slice1: on left side
bread_slice2: on right side
bacon: on left side
cheese: on right side
tomato: on right side
cucumber: on right side
ham: on left side
beef_patty: on left side
[Robot States]
1. Chad's gripper is empty
2. Dave's gripper is empty

[Target State]
[Food States]
bread_slice1: atop cutting_board
bread_slice2: on right side
bacon: on left side
cheese: atop ham
tomato: on right side
cucumber: on right side
ham: atop bread_slice1
beef_patty: on left side
[Robot States]
1. Chad's gripper is empty
2. Dave's gripper is empty

[Action]
EXECUTE\nNAME Chad ACTION PICK cheese \nNAME Dave ACTION PICK bread_slice1\n
"""

        return {'plan': plan_example}

    def get_initial_information(self):
        return "Recipe Order is" + str(self.recipe_order)


if __name__ == "__main__":
    env = MakeSandwichTask()
    test_obs = env.reset()
    print(env.describe_obs(test_obs))
    print(env.get_agent_prompt(test_obs, "Dave"))
    print(env.get_agent_prompt(test_obs, "Chad"))
