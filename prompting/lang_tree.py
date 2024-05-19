import torch
import numpy as np


def preprocess_action(action):
    action = action.replace("'''", "")
    action = action.replace("```", "")
    for i in action:
        if i == '\n':
            action = action[1:]
        else:
            break
    for i in reversed(action):
        if i == '\n':
            action = action[:-1]
        else:
            break

    action = action + '\n'

    return action


def preprocess_state(state):
    state = state.replace("'''", "")
    state = state.replace("```", "")
    state = state.replace('[Prediction] Begin', '')
    state = state.replace('[Prediction] End', '')
    for i in state:
        if i == '\n':
            state = state[1:]
        else:
            break

    for i in reversed(state):
        if i == '\n':
            state = state[:-1]
        else:
            break

    state = state + '\n'

    return state


class Nodes:
    def __init__(self, state):
        state = preprocess_state(state)

        self.state = state
        self.value = -1
        self.children = []
        self.avail_actions = []
        self.gamma = 0.995

    @property
    def num_children(self):
        return len(self.children)

    @property
    def num_avail_actions(self):
        return len(self.avail_actions)

    def add_child(self, action, child, virtual=False):
        if not virtual:
            action = preprocess_action(action)

        if action not in self.avail_actions:
            self.avail_actions.append(action)
            self.children.append({
                'action': action,
                'child': child
            })
        else:
            for temp_child in self.children:
                if temp_child['action'] == action:
                    temp_child['child'] = child
                    break

    def remove_child(self, action, virtual=False):
        if not virtual:
            action = preprocess_action(action)
        else:
            action = 'action_virtual'

        for i in range(len(self.children)):
            if self.children[i]['action'] == action:
                self.children.pop(i)
                break

        for i in range(len(self.avail_actions)):
            if self.avail_actions[i] == action:
                self.avail_actions.pop(i)
                break

    def update_child(self, action, child):
        action = preprocess_action(action)

        for i in range(len(self.children)):
            if self.children[i]['action'] == action:
                self.children[i]['child'] = child
                break

    def get_children(self):
        return self.children

    def get_child(self, action):
        for child in self.children:
            if child['action'] == action:
                return child['child']

        return None

    def compute_distance(self, target_state):
        pass

    def get_best(self, with_ucb=False, without_virtual=False):
        values = {}
        for child in self.children:
            if with_ucb:
                assert hasattr(child['child'], 'ucb_value'), 'The child node does not have ucb_value attribute'
                values[child['action']] = child['child'].ucb_value
            else:
                values[child['action']] = child['child'].value

        if without_virtual:
            values.pop('action_virtual')

        best_action = max(values, key=values.get)
        for child in self.children:
            if child['action'] == best_action:
                best_child = child['child']
                break

        return best_action, best_child

    def get_children_actions(self, without_virtual=False):
        actions = []
        for child in self.children:
            if without_virtual and child['action'] == 'action_virtual':
                continue
            else:
                actions.append(child['action'])

        return actions

    def compute_value(self, with_virtual=False):
        if with_virtual:
            if self.num_children == 1:
                return 0
            else:
                value = -1
                for child in self.children:
                    if child['action'] == 'action_virtual':
                        continue
                    else:
                        temp_value = child['child'].value * 0.995
                        if temp_value > value:
                            value = temp_value

                return value
        else:
            if self.num_children == 0:
                return None
            else:
                value = -1
                for child in self.children:
                    temp_value = child['child'].value * 0.995
                    if temp_value > value:
                        value = temp_value

                return value


class LangTree:
    def __init__(self, load=False, if_temp_tree=False, tree_path='data/lang_tree.pth'):
        self.nodes = []
        self.if_temp_tree = if_temp_tree
        self.tree_path = tree_path
        if load:
            self.load(tree_path)

        self.refine_tree()

    def refine_tree(self):
        for node in self.nodes:
            for child in node.children:
                if child['child'] not in self.nodes:
                    self.nodes.append(child['child'])
                    print(f'Node {child["child"].state} added to the tree')

        self.save()

    def init_values(self):
        for node in self.nodes:
            node.value = 0
            node.ucb_value = 0
            node.visit_num = 0

    def save(self):
        torch.save(self, self.tree_path)

    def load(self, path='data/lang_tree.pth'):
        try:
            temp_tree = torch.load(path)
            self.nodes = temp_tree.nodes
        except:
            print('No saved tree found')

    def update(self, transition):
        state, action, next_state = transition
        state_node = self.build_node(state)
        next_state_node = self.build_node(next_state)
        state_node.add_child(action, next_state_node)
        if not self.if_temp_tree:
            self.save()

    def build_virtual_node(self, state):
        assert self.if_temp_tree, 'This function is only for temporary tree'
        state_node = self.build_node(state)
        build_flag = not state_node.action_exhausted
        for child in state_node.children:
            if child['action'] == 'action_virtual':
                build_flag = False
                break

        if build_flag:
            action = 'action_virtual'
            next_state = state + '_sub_virtual'
            next_state_node = self.build_node(next_state)
            next_state_node.visit_num = 2
            next_state_node.value = -0.1
            state_node.add_child(action, next_state_node, virtual=True)

    def remove_node(self, state, action, virtual=False):
        state_node = self.get_node(state)
        next_state_node = state_node.get_child(action)
        state_node.remove_child(action, virtual=virtual)
        if next_state_node is not None:
            # I need to check if there exist other nodes that have next_state_node as child
            remove_flag = True
            for node in self.nodes:
                for child in node.children:
                    if child['child'] == next_state_node:
                        remove_flag = False
                        break

            if remove_flag:
                self.nodes.pop(self.nodes.index(next_state_node))

        if not self.if_temp_tree:
            self.save()

    def get_node(self, state):
        state = preprocess_state(state)

        for node in self.nodes:
            if node.state == state:
                return node

        return None

    def build_node(self, state):
        state = preprocess_state(state)

        state_exist = False
        for node in self.nodes:
            if node.state == state:
                state_exist = True
                break

        if not state_exist:
            node = Nodes(state)
            if self.if_temp_tree:
                node.action_exhausted = False
                node.value = 0
                node.ucb_value = 0
                node.visit_num = 0

            self.nodes.append(node)

        return node

    def check_transition_existence(self, transition):
        state, action, _ = transition
        state_node = self.build_node(state)
        for i in range(len(state_node.children)):
            if state_node.children[i]['action'] == action:
                return True

        return False

    def check(self, transition):
        state, action, next_state = transition
        state_node = self.build_node(state)
        next_state_node = self.build_node(next_state)
        for i in range(len(state_node.children)):
            if state_node.children[i]['action'] == action:
                if state_node.children[i]['child'].state != next_state:
                    state_node.children[i]['child'] = next_state_node
                    return False, {
                        'predicted_transition': [state, action, state_node.children[i]['child'].state],
                        'true_transition': transition
                    }
                else:
                    return True, None

        # if the program reaches here, it means that the action does not exist in the tree
        raise ValueError('Action does not exist in the tree')

    def compute_values(self, target_state, loop=5):
        # initial the valuse of all nodes to 0 except target state
        target_state = preprocess_state(target_state)
        state_exist = False
        for node in self.nodes:
            if node.state == target_state:
                state_exist = True
                node.value = 1
            else:
                node.value = 0

        if not state_exist:
            raise ValueError('Target state does not exist in the tree')

        for _ in range(loop):
            for node in self.nodes:
                if node.state != target_state:
                    temp_value = node.compute_value()
                    if temp_value is not None:
                        node.value = temp_value

    def compute_ucb_values(self, loop=5):
        for _ in range(loop):
            for node in self.nodes:
                temp_value = node.compute_value()
                if temp_value is not None:
                    node.value = temp_value

        for node in self.nodes:
            if node.visit_num == 0:
                raise ValueError('The visit number of the node is not 0')
            if node.num_children > 0:
                total_num = 0
                for child in node.children:
                    child_node = child['child']
                    total_num += child_node.visit_num

                for child in node.children:
                    child_node = child['child']
                    child_node.ucb_value = child_node.value / child_node.visit_num + 2 * np.sqrt(
                        np.log(total_num) / child_node.visit_num)
            # node.ucb_value = node.value / node.visit_num + 2 * np.sqrt(np.log(node.visit_num) / node.visit_num)

    def if_same(self, state, target_state):
        state = preprocess_state(state)
        target_state = preprocess_state(target_state)

        return state == target_state


class Tree2HtmlTranslator:
    def __init__(self, tree: LangTree):
        self.tree = tree
        self.html = ''
        self.nodes_str = ''
        self.edges_str = ''
        self.edges = []

    def get_default_header(self):
        self.html += f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Tutorial Demo</title>
  </head>
  <body>
    <div id="mountNode"></div>
    <script src="https://gw.alipayobjects.com/os/antv/pkg/_antv.g6-3.7.1/dist/g6.min.js"></script>
    <!-- 4.x and later versions -->
    <!-- <script src="https://gw.alipayobjects.com/os/lib/antv/g6/4.3.11/dist/g6.min.js"></script> -->
    <script>
      const graph = new G6.Graph({{
        container: 'mountNode',
        width: 1600,
        height: 3200,
        defaultNode: {{
          labelCfg: {{
            style: {{
              fill: '#fff',
            }},
          }},
        }},
        defaultEdge: {{
          labelCfg: {{
            autoRotate: true,
          }},
        }},
        nodeStateStyles: {{
          hover: {{
            fill: 'lightsteelblue',
          }},
          click: {{
            stroke: '#000',
            lineWidth: 3,
          }},
        }},
        edgeStateStyles: {{
          click: {{
            stroke: 'steelblue',
          }},
        }},
        layout: {{
          type: 'force',
          linkDistance: 100,
          preventOverlap: true,
          nodeStrength: -30,
          edgeStrength: 0.1,
        }},
        modes: {{
          default: ['drag-canvas', 'zoom-canvas', 'drag-node'],
        }},
      }});

      const main = async () => {{
        const remoteData = {{
"""

    def get_default_footer(self):
        self.html += f"""
        const nodes = remoteData.nodes;
        const edges = remoteData.edges;
        nodes.forEach((node) => {{
          if (!node.style) {{
            node.style = {{}};
          }}
          node.style.lineWidth = 1;
          node.style.stroke = '#666';
          node.style.fill = 'steelblue';
          switch (node.class) {{
            case 'state': {{
              node.type = 'rect';
              node.size = [180, 200];
              break;
            }}
            case 'c1': {{
              node.type = 'rect';
              node.size = [35, 20];
              break;
            }}
          }}
        }});
        edges.forEach((edge) => {{
          if (!edge.style) {{
            edge.style = {{}};
          }}
          edge.style.lineWidth = edge.weight;
          edge.style.opacity = 0.6;
          edge.style.stroke = 'grey';
        }});

        graph.data(remoteData);
        graph.render();

        graph.on('node:mouseenter', (e) => {{
          const nodeItem = e.item;
          graph.setItemState(nodeItem, 'hover', true);
        }});
        graph.on('node:mouseleave', (e) => {{
          const nodeItem = e.item;
          graph.setItemState(nodeItem, 'hover', false);
        }});
        graph.on('node:click', (e) => {{
          const clickNodes = graph.findAllByState('node', 'click');
          clickNodes.forEach((cn) => {{
            graph.setItemState(cn, 'click', false);
          }});
          const nodeItem = e.item;
          graph.setItemState(nodeItem, 'click', true);
        }});
        graph.on('edge:click', (e) => {{
          const clickEdges = graph.findAllByState('edge', 'click');
          clickEdges.forEach((ce) => {{
            graph.setItemState(ce, 'click', false);
          }});
          const edgeItem = e.item;
          graph.setItemState(edgeItem, 'click', true);
        }});
      }};
      main();
    </script>
  </body>
</html>
"""

    def process_tree(self):
        for i, node in enumerate(self.tree.nodes):
            node.id = i

        for node in self.tree.nodes:
            state = node.state.replace('\n', '\\n')
            self.nodes_str += f"""    {{"id": "{node.id}", "label": "{state}", "class": "state"}},\n"""
            for child in node.children:
                self.edges.append((node.id, child['child'].id))
                action = child['action'].replace('\n', '\\n')
                self.edges_str += f"""    {{"source": "{node.id}", "target": "{child['child'].id}", "label": "{action}", "weight": 1.0}},\n"""

        self.nodes_str = '    "nodes": [\n' + self.nodes_str + '    ],\n'
        self.edges_str = '    "edges": [\n' + self.edges_str + '    ]\n'

    def translate(self):
        self.get_default_header()
        self.process_tree()
        self.html += self.nodes_str
        self.html += self.edges_str
        self.html += '}\n'
        self.get_default_footer()
        with open('tree.html', 'w') as f:
            f.write(self.html)
