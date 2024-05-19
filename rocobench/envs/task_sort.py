import re
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pyquaternion import Quaternion
from rocobench.envs.base_env import MujocoSimEnv, EnvState
from rocobench.envs.robot import SimRobot
from rocobench.envs.constants import UR5E_ROBOTIQ_CONSTANTS, UR5E_SUCTION_CONSTANTS, PANDA_CONSTANTS

SORT_ALL_OBJECTS = [
    "panel2",
    "panel4",
    "panel6",
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]
ONE_OBJ_EACH = [
    "blue_square",
    "pink_polygon",
    "yellow_trapezoid",
]
SORTING_BIN_NAMES = [
    "panel2",
    "panel4",
    "panel6",
]

SORT_TASK_CONTEXT = """ 
7 panels on the table, ordered left to right: panel1,...,panel7. They form a straight assembly line, panel1 is closed to panel2 and farthest from panel7.
there are 3 cubes, each robot must place their cube on the correct target, their (cube, target_panel) pairs: 
alice: (blue_square, panel2), 
bob: (pink_polygon, panel4), 
chad: (yellow_trapezoid, panel6).
there are 3 robots, each with a limited reach range, this means they can only pick cubes from these panels, and can only place cubes on these panels. The (robot, [reachable_panels]) pairs: 
(alice, [panel1, panel2, panel3])
(bob, [panel3, panel4, panel5])
(chad, [panel5, panel6, panel7])
"""

SORT_TASK_DIALOG_PROMPT = ""

SORT_TASK_CHAT_PROMPT = """Robots discuss to find the best strategy. When each robot talk, it reasons about its own capability (e.g. I am Bob, I can only reach panel3), verifies other robots' claims (e.g. I am Alice, this trapezoid cube is indeed too far for Chad), then decide how to best achieve the task and help each other (e.g. therefore I will move it to panel3 so it's closer to Chad). 
carefully analyze Environment Feedback, Scene Description and others' responses to coordinate together. They talk in order [Alice],[Bob],[Chad],[Alice] ..., then, after everyone agreed, propose **exactly** one ACTION per robot, then stop talking. 
their entire chat history and the final plan are: 
"""

SORT_TASK_PLAN_PROMPT = """
Think step-by-step and reason about the best strategy for each robot to achieve their goal or best help others. Carefully consider Environment Feedback and Scene Description.
decide which cubes and panels can be reached by each robot. At each round, plan **exactly** one ACTION per robot. 
"""

SORTING_ACTION_SPACE = """[Action Space Definition]
Detail the action space for each robot with the stipulated actions: PICK PLACE, WAIT. All robots are executing actions simultaneously, with no distinction in the order of their actions.
- PICK <object> PLACE <target>: Execute if the gripper is empty. Pick up <object> and place it onto <target>, where <object> is a cube and <target> is a panel
- WAIT: Do nothing.

[Action Template]
EXECUTE
NAME Alice ACTION <action>
NAME Bob ACTION <action>
NAME Chad ACTION <action>

[Action Output Instructions]
1. Commence the output with 'EXECUTE\n'. Follow with one distinct action per robot on separate lines. 
2. Alice's action should be listed first, followed by Bob's action, and then Chad's action.
3. Agents' reach ranges are as follows:   
  + Alice: zone1=[panel1, panel2, panel3]
  + Bob: zone2=[panel3, panel4, panel5]
  + Chad: zone3=[panel5, panel6, panel7]

[Action Planning Instructions]
Constraints: A robot can only do one action at a time. Do not choose the joint action listed in the [Forbidden Actions]. A robot can only PICK and PLACE items in its reach range specified in [Robot Reach Range], DO NOT try to PICK and PLACE items outside reach range. There can only be up to one object on a panel. Remember that all robots are executing actions simultaneously, with no distinction in the order of their actions.
+ [Tip 1]: if there is a cube on intersection (panel3 or panel5). You have to first move it to other empty panel (like panel1, panel7 ...). Note that the target panel must be empty.
  - For example, if blue_square is on panel3 and you want to move pink_polygon from panel2 to panel4, you can first move blue_square to panel1 first and then move pink_polygon to panel3 and finally to panel4, thus avoiding collision.
+ [Tip 2]: move the objects to another zone through multiple movements by placing cubes at the intersection (panel3 and panel5). 
+ [Tip 3]: do not choose actions for each agent separately, and plan the action jointly by considering the multi-agent system as a whole.
+ [Tip 4]: try to merge the actions in [Forbidden Actions] to reduce the number of steps, but do not break the action constraints.
Please output your thinking process step-by-step. You have to follow these steps one by one to plan the [Action]: [a. Recall], [b. Action Plan], and [c. Pre Action Conclusion].
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions to not omit any steps.
[a. Recall] repeat the state as follows: [Repeat current State] and [Repeat target State]. 
  + [Repeat current State]: [blue_square: [], pink_polygon: [], yellow_trapezoid: []]
  + [Repeat target State]: [blue_square: [], pink_polygon: [], yellow_trapezoid: []]
[b. Action Plan] follow these instructions to plan the [Action] of the multi-agent system and list the thinking process step-by-step: [Interactive Cubes], [Available Targets], [Tip Recall], and [Plan Process].
  + [Interactive Cubes]
    - repeat the zones' range: [zone1=[panel...], ..]. initial three empty pools: [pool1={{}}, pool2={{}}, pool3={{}}].
    - for each cube, get the zones it belongs to, and add it into the pools: [cube -> state -> zone, pool=[pool1=[], pool2=[], ...]], e.g. [blue_square -> panel3 -> zone1&2, [pool1={{blue_square}}, pool2={{blue_square}}, pool3={{}}] ]
    - repeat the pools: [pool1, ...]. for each robot, list the cubes in its zone: [robot -> pool -> cubes] (Alice<->pool1, Bob<->pool2, Chad<->pool3) ... only be listed in this list, the robot can pick the cubes. 
  + [Occupied Panels]: list the occupied panels: []. the occupied panels cannot be robots' target, unless the cube on it will be picked by one robot in your planning action. 
  + [Available Targets]: for each robot, repeat its reach range: [robot: zone=[panel ...]]. only be listed in this list, the robot can place cubes on it.
  + [Tip Recall] Think about if the tips is available in this scenario one-by-on: []. Focus more on Tip 1 and its example, it is helpful for difficult tasks.
    - [tip 1: ..., tip 2: ...]
  + [Plan Process]
    - Repeat [Interactive Cubes]: [robot -> cubes, ...]
    - Repeat [Available Targets] by removing the panels in [Occupied Panels]: [robot -> panels, ...]
    - For each robot, the picked item cannot out of the above [Interactive Cubes] and the target of its action cannot out of the above [Available Targets]. Now, follow [Action Output Instructions] to plan an [Action] without breaking [Detailed Constraints for Action]: []
  + What is forbidden are the joint actions of the multi-agent system, the action of each agent that appears in these joint actions can be freely combined. 
    - For example, if the forbidden joint action is [A, B], then the agents can perform [A] and [B] separately like [A, C] or [D, B], but they cannot perform [A] and [B] at the same time. 
  + The WAIT is a good choice for you to handle the scenario where some actions are forbidden. But note that the WAIT action cannot be performed by all robots at the same time. 
[c. Pre Action Conclude] Conclude the [Action] according to the [Action Planning] in the format of [Detailed Constraints for Action]. Conclusion should be in the order Alice, Bob, Chad.

[Detailed Constraints for Action]
[Robot Reach Range] Before showing the constraints, you should note that each robot has its own reach range, and it cannot PICK or PLACE items outside its reach range.
  + Alice: zone1=[panel1, panel2, panel3]
  + Bob: zone2=[panel3, panel4, panel5]
  + Chad: zone3=[panel5, panel6, panel7]
[Detailed Constraints] Then, you should follow the constraints one-by-one and step-by-step to check if the action is correct: [Basic Constraints] and [PICK & PLACE Constraints]. You have to follow these constraints strictly and do not have your own understanding of the constraints.
Remember that all robots are executing actions simultaneously, with no distinction in the order of their actions.
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content.
1. [Basic Constraints] Output "[Basic Check]" and do as follows:
  + List the lines of the actions: []. Count the number: []. There must be three lines of actions. Check if it is equal to three: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + List the agents that perform WAIT: []. Count the number: []. The agents cannot all perform WAIT. Check if it is less than or equal to three: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + List the agents that perform PICK PLACE: []. List the <object> of PICK: []. Agents are not allowed to PICK <object> other than blue_square, pink_polygon, yellow_trapezoid. Check if all the <object> is one of blue_square, pink_polygon, yellow_trapezoid: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + List the agents that perform PICK PLACE: []. List the <object> of PICK: []. Agents are not allowed to PICK the same <object> at the same time. Check if the agents have the same <object> with each other: [yes/no]. If no, action is correct; otherwise incorrect. Conclusion: [].
  + List the agents that perform PICK PLACE: []. List the <target> of PLACE: []. Agents are not allowed to PLACE objects on the same panel. Check if the agents have the same <target> with each other: [yes/no]. If no, action is correct; otherwise incorrect. Conclusion: [].
2. [PICK & PLACE Constraints] Now, output "[PICK & PLACE Check]" and do as follows. All of this is [applicable], do not have your own understanding of the constraints.:
  + For simplicity, we use [agent] to denote the agent's name: [Alice, Bob, Chad]. When checking, please replace [agent] with the real agent's name.
    - List [agent]'s action: []. If it is PICK PLACE, repeat [agent]'s reach range: [panel...] and follow the instructions step by step: [PICK Range Check], [PLACE Range Check], and [PLACE Target Check].
      a. [PICK Range Check] List the <object> of PICK: [], and the <location> of this <object> in the current state: []. Agents are not allowed to PICK objects out of its reach range. Check if the <location> is in the reach range ([agent]'s: [panel...]): [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
      b. [PLACE Range Check] List the <target> of PLACE: []. Agents are not allowed to PLACE objects out of its reach range. Check if the <target> is in the reach range ([agent]'s: [panel...]): [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
      c. [PLACE Target Check] List the <target> of PLACE: []. Agents are not allowed to PLACE objects on the panel that already has an object except it is PICKed by one of the agents at the current timestep. Follow this instruction:
        - List the objects 'is on <target>' in the current state: pool1=[]. List the PICKed objects in the current state: pool2=[]. List the intersection of pool1 and pool2: pool3=[]. If pool3 is not empty, remove the objects in pool3 from pool1: pool1(after removal)=[]. Check if the pool1(after removal) is empty: [yes/no]. If yes, action is correct; otherwise incorrect. Conclusion: [].
  + For example: List Alice's action ... List Bob's action ... List Chad's action ...

[Example Actions]
Example: 'EXECUTE\nNAME Alice ACTION PICK blue_square PLACE panel3\nNAME Bob ACTION WAIT\nNAME Chad ACTION PICK yellow_trapezoid PLACE panel6\n'
"""

SORT_PREDICTING_CONSTRAINTS = f"""[Detailed Constraints for Prediction]
Check the following constraints and Fill in blanks in '[]'. Check these constraints one by one: 1, 2... Only get your conclusion according to the results of these checks! 
* Get [Interaction Item Pool]: list all PICKed objects in the [Action]: <pool>=[].
1. [Header Check] A header titled [State] -> A header titled [Cube States] -> List the Cube States
2. [Cube Order Check] Cubes must be listed in the following order: [blue_square, pink_polygon, yellow_trapezoid]
3. [Cube Format Check] Each line should be formatted as "<cube_name> is on <panel_name>". Three cube lines in total
4. [Cube State Check] Repeat cubes' locations: []. Don't have two cubes on the same panel
5. [PICK & PLACE Rule Check] For PICK & PLACE, repeat <object>=[], <target>=[], state of <object> in the prediction: [], it should be '<object> is on <target>': []
* When checking, you should repeat the prompt and fill in blanks, replace <obj> and <agent> with their true values, and give the line conclusion (yes/no/not applicable). For example:
4. [Cube State Check] Repeat cubes' locations: [panel1, panel2, panel3]. Don't have two cubes on the same panel: [yes]. Line conclusion: [yes]
* Constraint 5 may have multiple agents of checking. If so, you should output results separately. For example:
5.1 [PICK & PLACE Rule Check] For Alice's action: [PICK blue_square PLACE panel2], repeat <object>=[blue_square], <target>=[panel2], state of <object> in the prediction: [blue_square is on panel2], it should be '<object> is on <target>': [yes]. Line conclusion: [yes]
5.2 [PICK & PLACE Rule Check] For Bob's action: ... 5.3 ...
"""

SORT_ACTION_CONSTRAINTS = f"""[Detailed Constraints for Action]
Check the following constraints and Fill in blanks in '[]'. Check these constraints one by one: 1, 2... Do not forget any constraints!
* Get PICKed objects: <pool1>=[]
1. [Line Number Check] The line of the actions should be equal to three
2. [Agent Order Check] The order of the agents' actions should be Alice, Bob, Chad
3. [WAIT Number Check] The number of WAIT actions should be less than or equal to two
4. [PICK Object Check] Repeat <pool1>=[], there should not have duplicate objects: []
5. [PLACE Target Check] Get all PLACE target panels: [], there should not have duplicate targets: []
6. [PICK & PLACE Constraints] For simplicity, we use <agent> to denote the agent's name: [Alice, Bob, Chad]. When checking, please replace <agent> with the real agent's name.
  + If <agent>'s action is PICK PLACE: [], repeat <agent>'s reach range: <zone>-<agent>=[panel...]
  a. [PICK Range Check] Repeat <object>: [], location of <object>: <location>=[]. <location> should in <zone>-<agent>: []
  b. [PLACE Range Check] Repeat <target>: []. <target> should in the <zone>-<agent>: []
  c. [PLACE Target Check] Repeat <target>: [], object whose state is 'is on <target>': <X-object>=[], <pool1>=[blue_square, pink_polygon]. <X-object> should be empty: [] or in <pool1>: []
* When checking, you should repeat the prompt and fill in blanks and give the line conclusion. For example:
3. [WAIT Number Check] The number of WAIT actions should be less than or equal to two: [yes]. Line conclusion: [yes]
4. [PICK Object Check] Repeat <pool1>=[blue_square, pink_polygon], there should no duplicate objects: [yes]. Line conclusion: [yes]
* Constraint 6 may have multiple agents of checking. If so, you should output results separately. For example: 
6.1 [PICK & PLACE Constraints] For Alice's action: [PICK blue_square PLACE panel2], repeat Alice's reach range: <zone>-Alice=[panel1, panel2, panel3]
6.1.a ... 6.1.b [PLACE Range Check] Repeat <target>: [panel2]. <target> should in the <zone>-Alice: [yes]. Line conclusion: [yes]
6.1.c [PLACE Target Check] Repeat <target>: [panel2], object whose state is 'is on <target>': <X-object>=[pink_polygon], <pool1>=[blue_square, pink_polygon]. <X-object> should be empty: [no] or in <pool1>: [yes]. Line conclusion: [yes]
6.2 [PICK & PLACE Constraints] For Bob's action: ... 6.3 ...
"""

SORTING_STATE_SPACE = """[State Space Definition] begin
Define the state of the multi-agent system, which is composed of one category: [Cube States].
1. [Cube States]: Describe the status of the following three cube items: blue_square, pink_polygon, yellow_trapezoid.
2. The cube items must be listed in this order: blue_square, pink_polygon, yellow_trapezoid.

[State Template]
[State]
[Cube States]
blue_square is on <location>
pink_polygon is on <location>
yellow_trapezoid is on <location>

[State Output Instructions]
1. The [State] should follow the following formatting requirements:
  1) A header titled [State]. 
  2) A header titled [Cube States]. Then, list the Cube States. 
    + Each line represents the state of one cube item.
    + The cube items must be listed in this order: blue_square, pink_polygon, yellow_trapezoid.
    + Each line should be formatted as follows: "<item_name> is on <panel_name>", for example, "blue_square is on panel1". Don't substitute "on" with "in".

[State Prediction Instructions]
1. List Alice's action: []. List the [Cube States] after Alice's action: [].
2. List Bob's action: []. List the [Cube States] after Bob's action: [].
3. List Chad's action: []. List the [Cube States] after Chad's action: [].
4. Conclude the [Cube States] predicting result according to the format specified in [State Output Instructions].
5. Conclude the [State] in the format of [Detailed Constraints for State].

[Detailed Constraints for State]
You should specify the [State] according to the following constrains. A state is reasonable if it does not break the constraints one-by-one and step-by-step.
1. List the lines of [Cube States]: []. Count the number: [], and check if it is equal to three. There must be three lines of [Cube States].
2. List the cube items and their locations: []. Check if their locations are one of the following terms: 
  + on panel1
  + on panel2
  + on panel3
  + on panel4
  + on panel5
  + on panel6
  + on panel7
3. List the locations of the cubes: []. It is not allowed to have two cubes on the same panel. Check if they satisfy constraints.
3. List the locations of the cubes: []. The panel name should not contain " " in the middle. For example, use "panel2" instead of "panel 2". Check if they satisfy constraints.

Note: The system includes three cube items and seven panels by default. 

[Example States]
Example#1:
[State]
[Cube States]
blue_square is on panel3
pink_polygon is on panel7
yellow_trapezoid is on panel1
"""

prediction_example = f"""You can take the [Prediction Example] as a reference. This example only contains the outputted state.
[Prediction Example]
Example#1: 
[State]
[Cube States]
blue_square is on panel7
pink_polygon is on panel1
yellow_trapezoid is on panel3

[Action]
EXECUTE
NAME Alice ACTION PICK pink_polygon PLACE panel3
NAME Bob ACTION WAIT
NAME Chad ACTION WAIT

[Prediction]
[Cube States]
blue_square is on panel7
pink_polygon is on panel3
yellow_trapezoid is on panel3
"""

SORTING_PREDICTING_PROMPT = f"""[Predicting Instruction]
You will be provided with the [State] and the [Action] of the multi-agent system. You should think step by step to output the [Prediction] of the next [State] based on the given [State] and [Action]. The format of the [Prediction] should follow the [Detailed Constraints for State]
Please output your thinking process step-by-step by following theses steps: [Interaction Item Pool], [Action Forward Rule] and [Prediction Conclusion]
The most important thing: Follow the instructions step-by-step and ensure each step is completed precisely. Repeat the instructions and fill in the blanks '[]' without introducing any modifications or additional content
1. You should think step by step, and make sure the [State] does not break the [Detailed Constraints for State].
3. You should follow the format described in [Detailed Constraints for State].
4. You should output your thinking process step-by-step in the following order:
  + [State Prediction]: Strictly follow the [State Prediction Instructions] to predict the [State] of the multi-agent system step by step and list the thinking process.
  + [State Conclusion]: Conclude the [State] in the format of [Detailed Constraints for State].
  + [State Constraints Check]: check if the [State] satisfies the [Detailed Constraints for State]. You should follow the steps in [Detailed Constraints for State] to check the [State] step by step.
  If the [State] didn't pass the [State Constraints Check]:
    - Follow these instructions to predict a new [State] step by step and list the thinking process:
      - Go back to a new [State Prediction] and [State Constraints Check] process again.
      - [State Constraints Check]: check if the [State] satisfies the [Detailed Constraints for State].
      - Re-predict the [State] until a new [State] that passes [State Constraints Check] is obtained. You can try 2-nd, 3-rd, 4-th ... Plan to get the final [State].
  + [State Output]: output the final state in the format of [Detailed Constraints for State].
{prediction_example}
"""

class SortOneBlockTask(MujocoSimEnv):
    def __init__(
            self,
            filepath: str = "rocobench/envs/task_sort.xml",
            one_obj_each: bool = False,
            **kwargs,
    ):
        self.robot_names = ["ur5e_robotiq", "panda", "ur5e_suction"]
        self.agent_names = ["Alice", "Bob", "Chad"]

        self.robot_name_map = {
            "ur5e_robotiq": "Alice",
            "panda": "Bob",
            "ur5e_suction": "Chad",
        }
        self.robot_name_map_inv = {
            "Alice": "ur5e_robotiq",
            "Bob": "panda",
            "Chad": "ur5e_suction",
        }
        self.robots = dict()
        self.obj_to_panel = dict()
        self.cube_names = ONE_OBJ_EACH
        self.cube_to_bin = dict(
            blue_square="panel2",
            pink_polygon="panel4",
            yellow_trapezoid="panel6",
        )
        with open('rocobench/envs/task_sort_cases.json', 'r') as f:
            self.cases_information = json.load(f)

        self.full_cases = {}
        for optimal_step in range(1, 7):
            self.full_cases[optimal_step] = []
            for case in self.cases_information:
                if case['optimal_steps'] == optimal_step:
                    self.full_cases[optimal_step].append(case['state'])

        self.cases = None
        self.set_optimal_steps(1)

        super(SortOneBlockTask, self).__init__(
            filepath=filepath,
            task_objects=SORT_ALL_OBJECTS,
            agent_configs=dict(
                ur5e_robotiq=UR5E_ROBOTIQ_CONSTANTS,
                panda=PANDA_CONSTANTS,
                ur5e_suction=UR5E_SUCTION_CONSTANTS,
            ),
            **kwargs
        )

        self.panel_coords = dict()
        self.name = 'sort_central'
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if 'panel' in geom.name:
                self.panel_coords[geom.name] = geom.pos

        self.robots[
            self.robot_name_map["ur5e_robotiq"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_ROBOTIQ_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["panda"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **PANDA_CONSTANTS,
        )
        self.robots[
            self.robot_name_map["ur5e_suction"]
        ] = SimRobot(
            physics=self.physics,
            use_ee_rest_quat=False,
            **UR5E_SUCTION_CONSTANTS,
        )

        self.align_threshold = 0.1
        self.bin_slot_pos = dict()
        for bin_name in SORTING_BIN_NAMES:
            for slot in ["middle"]:  # ["left", "middle", "right"]:
                self.bin_slot_pos[f"{bin_name}_{slot}"] = self.physics.named.data.site_xpos[f"{bin_name}_{slot}"]

        self.robot_to_bin = dict(
            ur5e_robotiq="panel2",
            panda="panel4",
            ur5e_suction="panel6",
        )
        self.bin_x_coords = {
            'panel2': -0.8,
            'panel4': 0,
            'panel6': 0.8,
        }

        self.cube_targets = dict(
            Alice=("blue_square", "panel2"),
            Bob=("pink_polygon", "panel4"),
            Chad=("yellow_trapezoid", "panel6"),
        )
        self.reachable_panels = dict(
            Alice=["panel1", "panel2", "panel3"],
            Bob=["panel3", "panel4", "panel5"],
            Chad=["panel5", "panel6", "panel7"],
        )

    @property
    def use_preplace(self):
        return True

    def set_optimal_steps(self, optimal_steps):
        self.optimal_steps = optimal_steps
        self.cases = self.full_cases[optimal_steps]
        self.case_number = len(self.cases)

    def get_action_prompt(self) -> str:
        return SORTING_ACTION_SPACE

    def get_action_space_prompt(self) -> str:
        return SORTING_ACTION_SPACE

    def get_state_space_prompt(self) -> str:
        return SORTING_STATE_SPACE

    def get_state_prompt(self, obs):
        cube_states = []
        for cube in self.cube_names:
            desp = self.describe_cube_state(obs, cube)
            cube_states.append(desp)
        cube_states = "\n".join(cube_states)

        # robot_names = self.robot_name_map_inv.values()
        # robot_states = {}
        # for robot_name in robot_names:
        #     agent_state = self.describe_robot_state(obs, robot_name)
        #     robot_states[robot_name] = agent_state

        state_prompt = f"""[State]
[Cube States]
{cube_states}
"""
        return state_prompt

    def get_robot_name(self, agent_name):
        return self.robot_name_map_inv[agent_name]

    def get_agent_name(self, robot_name):
        return self.robot_name_map[robot_name]

    def get_robot_config(self) -> Dict[str, Dict[str, Any]]:
        return self.agent_configs

    def get_sim_robots(self) -> Dict[str, SimRobot]:
        """NOTE this is indexed by agent name, not actual robot names"""
        return self.robots

    def get_robot_reach_range(self, robot_name: str) -> Dict[str, Tuple[float, float]]:
        if robot_name == "ur5e_robotiq" or robot_name == self.robot_name_map["ur5e_robotiq"]:
            return dict(x=(-1.4, -0.1), y=(0.1, 1.3), z=(0.16, 1))

        elif robot_name == "panda" or robot_name == self.robot_name_map["panda"]:
            return dict(x=(-0.7, 0.7), y=(-0.21, 1.3), z=(0.16, 1))

        elif robot_name == "ur5e_suction" or robot_name == self.robot_name_map["ur5e_suction"]:
            return dict(x=(0.2, 1.5), y=(0.1, 1.3), z=(0.16, 1))

        else:
            raise NotImplementedError

    def check_reach_range(self, robot_name, point: Tuple[float, float, float]) -> bool:
        reach_range = self.get_robot_reach_range(robot_name)
        for i, axis in enumerate(["x", "y", "z"]):
            if point[i] < reach_range[axis][0] or point[i] > reach_range[axis][1]:
                return False
        return True

    def sample_initial_scene(self, case_id: int = 0):
        # find the pre-defined panel positions in the xml
        initial_case_state = self.cases[case_id]
        case_state = {
            "blue_square": initial_case_state["A"],
            "pink_polygon": initial_case_state["B"],
            "yellow_trapezoid": initial_case_state["C"],
        }

        tosample_panels = []
        for n in range(self.physics.model.ngeom):
            geom = self.physics.model.geom(n)
            if geom.name in ['panel1', 'panel3', 'panel5', 'panel7']:
                tosample_panels.append(
                    (geom.name, geom.pos, geom.size)
                )

        target_panels = ['panel2', 'panel4', 'panel6']
        for panel in target_panels:
            tosample_panels.append(
                (panel, self.physics.data.geom(panel).xpos, geom.size)
            )

        assert len(tosample_panels) >= 3, "Not enough panel positions to sample from"

        far_panels = dict()
        far_panels['square'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] > 0.15]
        far_panels['polygon'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.7 or tup[1][0] > 0.9]
        far_panels['trapezoid'] = [i for i, tup in enumerate(tosample_panels) if tup[1][0] < -0.15]

        # sample the panel positions
        occupied_idxs = []
        for i, name in enumerate(ONE_OBJ_EACH):
            try:
                qpos_slice = self.physics.named.data.qpos._convert_key(f"{name}_joint")
            except KeyError:
                print('Skipping object: ', name, ' because its _joint does not exist in the xml file')
                continue
            assert int(qpos_slice.stop - qpos_slice.start) == 7, "object qpos must be 7-dim"
            start = qpos_slice.start
            stop = qpos_slice.stop
            shape = name.split('_')[1]

            # idx = self.random_state.choice(far_panels[shape])
            # # remove this index from the list of available panels
            # for shape, idxs in far_panels.items():
            #     if idx in idxs:
            #         idxs.remove(idx)

            object_state = 'panel{}'.format(case_state[name])
            for idx, (panel_name, panel_pos, panel_size) in enumerate(tosample_panels):
                if panel_name == object_state:
                    break

            if panel_name is None:
                raise ValueError(f"Panel not found for {name}")

            # panel_name, panel_pos, panel_size = tosample_panels[idx]
            self.obj_to_panel[name] = panel_name
            # sample a random position within the panel
            # new_pos = self.random_state.uniform(
            #     low=panel_pos - panel_size / 2 + 0.001,
            #     high=panel_pos + panel_size / 2 - 0.001,
            # )
            new_pos = panel_pos.copy()

            new_quat = Quaternion(
                axis=[0, 0, 1],
                angle=self.random_state.uniform(low=0, high=2 * np.pi)
            )
            new_quat = np.array([new_quat.w, new_quat.x, new_quat.y, new_quat.z])
            old_pos = self.physics.named.data.qpos[start:stop]
            new_pos[2] = old_pos[2]
            self.physics.named.data.qpos[start:stop] = np.concatenate([new_pos, new_quat])

        self.physics.forward()
        self.physics.step(100)

    def get_obs(self):
        obs = super().get_obs()
        for name in self.robot_names:
            assert getattr(obs, name) is not None, f"Robot {name} is not in the observation"
        return obs

    def describe_robot_state(self, obs: EnvState, robot_name: str):
        agent_name = self.get_agent_name(robot_name)
        robot_state = getattr(obs, robot_name)
        x, y, z = robot_state.ee_xpos

        dist_to_panels = [(name, np.linalg.norm(robot_state.ee_xpos[:2] - pos[:2])) for name, pos in
                          self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0]
        robot_desp = ""  # f"{agent_name}'s gripper is closest to {closest_panel}, "

        if len(robot_state.contacts) == 0:
            obj = "empty"
        else:
            obj = "holding " + ",".join([c for c in robot_state.contacts])
        # robot_desp += f"{agent_name}'s gripper is at ({x:.2f} {y:.2f} {z:.2f}), holding {obj},"
        robot_desp += f"{agent_name}'s gripper is {obj},"

        reachables = []
        not_reachables = []
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name]
            top_site = block_state.sites[f'{block_name}_top']
            if self.check_reach_range(robot_name, top_site.xpos):
                reachables.append(block_name)
            else:
                not_reachables.append(block_name)

        if len(reachables) > 0:
            robot_desp += f" can reach cubes: "
            for obj in reachables:
                robot_desp += f"{obj}, "
        if len(not_reachables) > 0:
            robot_desp += f"can't reach cubes: "
            for obj in not_reachables:
                robot_desp += f"{obj}, "

        return robot_desp

    def get_allowed_collision_pairs(self) -> List[Tuple[int, int]]:
        ret = []
        cube_ids = [self.physics.model.body(cube).id for cube in self.cube_names]

        table_id = self.physics.model.body("table").id
        bin_ids = [self.physics.model.body(bin_name).id for bin_name in SORTING_BIN_NAMES]

        for link_id in self.robots["Alice"].all_link_body_ids + self.robots["Bob"].all_link_body_ids + self.robots[
            "Chad"].all_link_body_ids:
            for cube_id in cube_ids:
                ret.append((link_id, cube_id))
            for bin_id in bin_ids:
                ret.append((link_id, bin_id))
            ret.append((link_id, table_id))

        for cube_id in cube_ids:
            ret.append((cube_id, table_id))
            for cube_id2 in cube_ids:
                if cube_id != cube_id2:
                    ret.append((cube_id, cube_id2))
            for bin_id in bin_ids:
                ret.append((cube_id, bin_id))

        return ret

    def get_cube_panel(self, obs, cube_name: str):
        cube_state = obs.objects[cube_name]
        dist_to_panels = [(name, np.linalg.norm(cube_state.xpos[:2] - pos[:2])) for name, pos in
                          self.panel_coords.items()]
        closest_panel = min(dist_to_panels, key=lambda x: x[1])[0]
        for pname in ["panel2", "panel4", "panel6"]:
            if pname in obs.objects[cube_name].contacts:
                closest_panel = pname
                break
        return closest_panel

    def describe_cube_state(self, obs: EnvState, cube_name: str):
        cube_state = obs.objects[cube_name]
        top_site = cube_state.sites[f'{cube_name}_top']
        x, y, z = top_site.xpos
        cube_desp = ""
        for slot_name, pos in self.bin_slot_pos.items():
            if np.linalg.norm(pos[:2] - cube_state.xpos[:2]) < self.align_threshold:
                slot_name = "_".join(slot_name.split("_")[:-1])
                cube_desp = f"{cube_name} is on {slot_name}"
                break
        if len(cube_desp) == 0:
            closest_panel = self.get_cube_panel(obs, cube_name)
            cube_desp = f"{cube_name} is on {closest_panel}"
        return cube_desp

    def describe_obs(self, obs: EnvState):
        """ For each cube, just describe whether it's on a bin, or between which two bins, no output numerical coordinates """
        object_desp = "[Scene description]\n"
        for cube_name in ONE_OBJ_EACH:
            object_desp += self.describe_cube_state(obs, cube_name) + "\n"

        robot_desp = ""
        # for robot_name, agent_name in self.robot_name_map.items():
        #     robot_desp += self.describe_robot_state(obs, robot_name)+"\n"
        # robot_desp = robot_desp[:-2]+".\n"
        full_desp = object_desp + robot_desp
        return full_desp

    def get_reward_done(self, obs):
        reward = 1
        done = True
        for block_name in ONE_OBJ_EACH:
            block_state = obs.objects[block_name]
            correct_bin = self.cube_to_bin[block_name]
            bin_pos = self.bin_slot_pos[f"{correct_bin}_middle"]
            if np.linalg.norm(bin_pos[:2] - block_state.xpos[:2]) > self.align_threshold and (
                    correct_bin not in obs.objects[block_name].contacts):
                reward = 0
                done = False
                break
        return reward, done

    def describe_task_context(self):
        return SORT_TASK_CONTEXT

    def get_target_prompt(self, obs: EnvState):
        # get the prompt of the final state
        # task specific!
        state_desp = self.get_state_prompt(obs)
        state_desp_dict = {}
        for cube_item in self.cube_names:
            re_match = re.search(f"{cube_item} is on panel\d", state_desp)
            if re_match is not None:
                state_desp_dict[cube_item] = f"{cube_item} is on {self.cube_to_bin[cube_item]}\n"
            else:
                raise ValueError('Cube item not on table')

        target_state_desp = f"""[State]
[Cube States]
"""
        for cube_item in self.cube_names:
            target_state_desp += state_desp_dict[cube_item]

        self.target_state_desp = target_state_desp

    def get_grasp_site(self, obj_name: str = "pink_polygon") -> str:
        return f"{obj_name}_top"

    def get_target_pos(self, agent_name, target_name, target_type: str = 'site') -> Optional[np.ndarray]:
        """ useful for parsing place targets """
        ret = None
        robot_name = self.robot_name_map_inv[agent_name]
        if 'panel' in target_name:
            try:
                ret = self.physics.data.geom(target_name).xpos.copy()
            except KeyError:
                return None

            if target_name == 'panel3':
                if 'panda' in robot_name:
                    ret[0] -= 0.12 # / 2
                    ret[1] -= 0.1 # / 2
                else:
                    ret[0] += 0.12 # / 2
                    ret[1] += 0.1 # / 2
            if target_name == 'panel5':
                if 'panda' in robot_name:
                    ret[0] += 0.12 # / 2
                    ret[1] -= 0.1 # / 2
                else:
                    ret[0] -= 0.12 # / 2
                    ret[1] += 0.1 # / 2

            ret[2] = 0.5
        elif target_name in self.cube_names:
            sname = f"{target_name}_top"
            ret = self.physics.data.site(sname).xpos.copy()
        else:
            ret = None
        # print(f"Agent: {agent_name} target site for {target_name} is {ret}")

        return ret

    def get_contact(self):
        contacts = super().get_contact()
        # temp fix!
        contacts["ur5e_robotiq"] = [c for c in contacts["ur5e_robotiq"] if c in self.cube_names]
        contacts["panda"] = [c for c in contacts["panda"] if c in self.cube_names]
        contacts["ur5e_suction"] = [c for c in contacts["ur5e_suction"] if c in self.cube_names]
        return contacts

    def get_task_feedback(self, obs, llm_plan, pose_dict):
        feedback = ""
        error_info = {
            'IK_error': False,
            'dynamics_error': False,
            'task_error': False
        }
        pick_items, place_targets = [], []
        for agent_name, action_str in llm_plan.action_strs.items():
            if 'PICK' in action_str and 'PLACE' in action_str:
                obj = action_str.split('PICK')[1].split('PLACE')[0].strip()
                target = action_str.split('PLACE')[1].strip()
                pick_items.append(obj)
                place_targets.append(target)

        for agent_name, action_str in llm_plan.action_strs.items():
            if ('PICK' in action_str and 'PLACE' not in action_str) or \
                    ('PLACE' in action_str and 'PICK' not in action_str):
                feedback += f"{agent_name}'s ACTION must contain both PICK and PLACE"
            if 'PICK' in action_str and 'PLACE' in action_str:
                obj = action_str.split('PICK')[1].split('PLACE')[0].strip()
                target = action_str.split('PLACE')[1].strip()
                if obj not in self.cube_names:
                    # check if the object is a cube
                    feedback += f"{agent_name} can't PICK {obj}, it's not a cube item."
                if self.get_cube_panel(obs, cube_name=obj) not in self.reachable_panels[agent_name]:
                    feedback += f"{agent_name} can't PICK {obj}, it's out of reach."
                if target not in self.reachable_panels[agent_name]:
                    feedback += f"{agent_name} can't PLACE {obj} on {target}, it's out of reach."
                # check if the target panel is occupied except the cube is picked by the agent
                for cube_name in self.cube_names:
                    if self.get_cube_panel(obs, cube_name) == target and cube_name not in pick_items:
                        feedback += f"{agent_name} can't PLACE {obj} on {target}, it's occupied."

        # check if the pick and place actions have the same object
        if len(set(pick_items)) != len(pick_items):
            feedback += "The same object can't be picked by multiple agents at the same time."
        if len(set(place_targets)) != len(place_targets):
            feedback += "The same target panel can't be placed by multiple agents at the same time."

        if all(['WAIT' in action_str for action_str in llm_plan.action_strs.values()]):
            feedback += f"You can't all WAIT. The task is not complete, at least one robot should be acting."

        if len(feedback) > 0:
            error_info['dynamics_error'] = True

        return feedback, error_info

    def get_object_joint_name(self, obj_name):
        return f"{obj_name}_joint"

    def chat_mode_prompt(self, chat_history: List[str] = []):
        return SORT_TASK_CHAT_PROMPT

    def central_plan_prompt(self):
        return SORT_TASK_PLAN_PROMPT

    def dialog_mode_prompt(self):
        return SORT_TASK_DIALOG_PROMPT

    def get_task_prompt(self):
        task_prompt = f"""[Task Description]
Task: Cooperative Sorting in a Multi-Agent System

Agents: Alice, Bob and Chad
Alice - Can only PICK and PLACE cube items on zone1=[panel1, panel2, panel3].
Bob - Can only PICK and PLACE cube items on zone2=[panel3, panel4, panel5].
Chad - Can only PICK and PLACE cube items on zone3=[panel5, panel6, panel7].

Objective: Collaboratively place the cubes on the panels as follows: place blue_square on panel2, place pink_polygon on panel4, place yellow_trapezoid on panel6. The cube items will be placed randomly on all panels of the table at the beginning of an episode. 

Instructions:
1. Agent can only perform actions described in [Action Space Definition]. Please do not imagine other actions.
2. Some items can be beyond the reach range of certain agents. In this case, other agents might do it a favor by moving the item closer to it.
3. This is a Markov Decision Process (MDP). There exists an external environment interacting with the agents. 

NOTICE!!! The robots DO NOT need to perform actions one by one. They can perform actions simultaneously, as long as the actions do not conflict with each other.

Constraints:
1. The state of the multi-agent system must adhere to the prescribed [State Space Definition]! 
2. Each agent selects one action from the provided [Action Space Definition]. 
"""
        return task_prompt

    def get_action_constrains(self, state_desp: str):
        cube_state_dict = {}
        state_lines = state_desp.split("\n")
        for line in state_lines:
            if "is on" in line:
                cube_name, panel_name = re.findall(r"(\w+) is on (\w+)", line)[0]
                cube_state_dict[cube_name] = panel_name

        reachable_cubes_dict = {}
        for agent_name in self.agent_names:
            reachable_cubes = ", ".join(
                [cube for cube, panel in cube_state_dict.items() if panel in self.reachable_panels[agent_name]]
            )
            reachable_cubes_dict[agent_name] = reachable_cubes

        reachable_cubes_prompt = f"[Reachable Cubes]"
        for agent_name in self.agent_names:
            reachable_cubes_prompt += f"\n{agent_name}: {reachable_cubes_dict[agent_name]}"

        prompt = SORT_ACTION_CONSTRAINTS
        prompt += reachable_cubes_prompt

        return prompt

    def get_check_tail(self):
        predicting_tail = f"""[Start]
Get [Interaction Item Pool]: list all ...
1. [Header Check]
2. [Cube Order Check]
...
[Conclusion] ..."""
        action_tail = f"""[Start]
* Get PICKed objects: <pool1>= ...
1. [Line Number Check]
2. [Agent Order Check]
...
[Conclusion] ..."""
        return {'predicting': predicting_tail, 'action': action_tail}

    def get_policy_tail(self):
        prompt = f"""
I am ..., and my target is to PUT ... on ..., and other robots ..., Since ..."""
        return prompt

    def if_fail(self, state_desp):
        return False

    def get_state_template(self):
        template = f"""[State]
[Cube States]
blue_square is on <panel_name>
pink_polygon is on <panel_name>
yellow_trapezoid is on <panel_name>"""

        return template

    def get_prediction_constrains(self):
        return SORT_PREDICTING_CONSTRAINTS

    def process_action(self, action):
        action_lines = action.split("\n")
        agent_action_dict = {}
        for line in action_lines:
            for agent_name in self.agent_names:
                if f'NAME {agent_name}' in line:
                    agent_action_dict[agent_name] = line

        processed_action = 'EXECUTE\n'
        for agent_name in self.agent_names:
            if agent_name in agent_action_dict.keys():
                processed_action += agent_action_dict[agent_name] + '\n'
            else:
                processed_action += f'NAME {agent_name} ACTION WAIT\n'

        if 'ERROR' in action:
            processed_action = 'ERROR\n'

        return processed_action

    def process_state(self, state):
        state_lines = state.split("\n")
        for i, line in enumerate(state_lines):
            if '[State]' in line:
                state_lines = state_lines[i:]
                break

        cube_states = {}
        for line in state_lines:
            for cube_name in self.cube_names:
                if cube_name + ' is on' in line:
                    cube_states[cube_name] = line

        processed_state = '[State]\n'
        processed_state += '[Cube States]\n'
        try:
            for cube_name in self.cube_names:
                processed_state += cube_states[cube_name] + '\n'
        except:
            processed_state = 'ERROR'

        return processed_state

    def get_repeat_template(self):
        state_template = f"""[blue_square: [], pink_polygon: [], yellow_trapezoid: []]"""
        action_template = f"""[Alice: [], Bob: [], Chad: []]"""

        return {'state': state_template, 'action': action_template}

    def get_initial_information(self):
        return 'nothing'

    def get_predicting_prompt(self):
        return SORTING_PREDICTING_PROMPT

    def get_examples(self):
        plan_example = f"""[Plan Example]
Example#1:
[Original State]
[State]
[Cube States]
blue_square is on panel1
pink_polygon is on panel4
yellow_trapezoid is on panel6

[Target State]
[Cube States]
blue_square is on panel2
pink_polygon is on panel4
yellow_trapezoid is on panel6

[Action]
EXECUTE
NAME Alice ACTION PICK blue_square PLACE panel2
NAME Bob ACTION WAIT
NAME Chad ACTION WAIT
"""

        return {'plan': plan_example}


if __name__ == "__main__":
    env = SortOneBlockTask(np_seed=0, render_point_cloud=0)
    test_obs = env.reset()
    # obs.scene.show()
    print(env.get_action_prompt())
    # print(env.get_agent_prompt(test_obs, "Alice"))
    breakpoint()
    img = env.physics.render(camera_id="teaser", height=480, width=600)

