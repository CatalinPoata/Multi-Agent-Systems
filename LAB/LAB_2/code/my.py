from copy import copy

from environment import *
from typing import List, Tuple
import time

class MyAgent(BlocksWorldAgent):

    def __init__(self, name: str, target_state: BlocksWorld):
        super(MyAgent, self).__init__(name=name)

        self.target_state = target_state

        """
        The agent's belief about the world state. Initially, the agent has no belief about the world state.
        """
        self.belief: BlocksWorld = None

        """
        The agent's current desire. It is expressed as a list of blocks for which the agent wants to make a plan to bring to their corresponding
        configuration in the target state. 
        The list can contain a single block or a sequence of blocks that represent: (i) a stack of blocks, (ii) a row of blocks (e.g. going level by level).
        """
        self.current_desire : List[Block] = None

        """
        The current intention is the agent plan (sequence of actions) that the agent is executing to achieve the current desire.
        """
        self.current_intention: List[BlocksWorldAction] = []


    def response(self, perception: BlocksWorldPerception) -> BlocksWorldAction:
        ## if the perceived state contains the target state, the agent has achieved its goal
        if perception.current_world.contains_world(self.target_state):
            return AgentCompleted()

        print(f"Intention before revision: {self.current_intention}")
        ## revise the agents beliefs based on the perceived state
        self.revise_beliefs(perception)
        print(f"Intention after revision: {self.current_intention}")

        ## Single minded agent intention execution: if the agent still has actions left in the current intention, and the intention
        ## is still applicable to the perceived state, the agent continues executing the intention
        print(f"Are there intentions left?: {len(self.current_intention)}")
        if len(self.current_intention) > 0:
            print(
                f"Is the first intention doable?: {self._can_apply_action(self.current_intention[0], perception.current_world, perception.holding_block)}")
        if len(self.current_intention) > 0 and self._can_apply_action(self.current_intention[0], perception.current_world, perception.holding_block):
            return self.current_intention.pop(0)
        else:
            ## the agent has to set a new current desire and plan to achieve it
            self.current_desire, self.current_intention = self.plan()

        print(f"Intention after plan: {self.current_intention}")

        ## If there is an action in the current intention, pop it and return it
        if len(self.current_intention) > 0:
            return self.current_intention.pop(0)
        else:
            ## If there is no action in the current intention, return a NoAction
            return NoAction()        


    def _can_apply_action(self, act: BlocksWorldAction, world: BlocksWorld, holding_block: str) -> bool:
        """
        Check if the action can be applied to the current world state.
        """
        ## create a clone of the world
        sim_world = world.clone()

        ## apply the action to the clone, surrpressing any exceptions
        try:
            ## locking can be performed at any time, so check if the action is a lock actio
            if act.get_type() == "lock":
                ## try to lock the block
                sim_world.lock(act.get_argument())
            else:
                if holding_block is None:
                    if act.get_type() == "putdown" or act.get_type() == "stack":
                        ## If we are not holding anything, we cannot putdown or stack a block
                        print("We're not holding shit!")
                        return False
                    
                    if act.get_type() == "pickup":
                        ## try to pickup the block
                        sim_world.pickup(act.get_argument())
                    elif act.get_type() == "unstack":
                        ## try to unstack the block
                        sim_world.unstack(act.get_first_arg(), act.get_second_arg())
                else:
                    ## we are holding a block, so we can only putdown or stack
                    if act.get_type() == "pickup" or act.get_type() == "unstack":
                        ## If we are holding a block, we cannot pickup or unstack
                        return False

                    if act.get_type() == "putdown":
                        print("We're not holding the right thing?????")
                        print(f"Holding block: {holding_block}")
                        print(f"Argument: {act.get_argument().__str__()}")
                        ## If we want to putdown the block we have to check if it's the same block we are holding
                        if act.get_argument() != holding_block:
                            return False

                    if act.get_type() == "stack":
                        ## If we want to stack the block we have to check if it's the same block we are holding
                        if act.get_first_arg() != holding_block:
                            return False
                        ## try to stack the block
                        sim_world.stack(act.get_first_arg(), act.get_second_arg())
        except Exception as e:
            return False
        
        return True

    def get_placed_blocks(self, perceived_world_state: BlocksWorld):
        placed_blocks = []
        for stack in perceived_world_state.get_stacks():
            for block in stack.get_blocks():
                placed_blocks.append(block)
        return placed_blocks

    def get_missing_blocks(self, perceived_world_state: BlocksWorld):
        missing_blocks = []
        placed_blocks = self.get_placed_blocks(perceived_world_state)
        for block in perceived_world_state.get_all_blocks():
            if block not in placed_blocks:
                missing_blocks.append(block)
        return missing_blocks

    def init_desires(self) -> List[Block]:
        ordered_blocks = []
        stacks = self.target_state.get_stacks()
        max_height = max(len(stack.get_blocks()) for stack in stacks)

        for level in range(max_height):
            for stack in stacks:
                blocks = stack.get_blocks()
                if level < len(blocks):
                    ordered_blocks.append(blocks[level])

        return ordered_blocks

    def get_target_block(self, current_block):
        for stack in self.target_state.get_stacks():
            if current_block in stack.get_blocks():
                return stack.get_below(current_block)

    def is_target_block_on_board(self, current_block):
        target_block = self.get_target_block(current_block)
        missing_blocks = self.get_missing_blocks(self.belief)
        if target_block in missing_blocks:
            return False
        return True

    def is_target_block_free(self, current_block):
        target_block = self.get_target_block(current_block)
        if target_block is None:
            return True
        if self.is_target_block_on_board(current_block):
            for stack in self.belief.get_stacks():
                if target_block in stack.get_blocks():
                    if stack.get_above(target_block) is None:
                        return True
        return False

    def is_block_in_position(self, current_block):
        if current_block is None:
            return True

        missing_blocks = self.get_missing_blocks(self.belief)
        if current_block in missing_blocks:
            return False

        current_stack = []
        target_stack = []

        for stack in self.target_state.get_stacks():
            if current_block in stack.get_blocks():
                target_stack.extend(stack.get_blocks())

        for stack in self.belief.get_stacks():
            if current_block in stack.get_blocks():
                current_stack.extend(stack.get_blocks())

        if target_stack.index(current_block) != current_stack.index(current_block):
            return False

        for i in range(target_stack.index(current_block) + 1):
            if target_stack[i] != current_stack[i]:
                return False
        return True

    def is_block_on_board(self, current_block):
        missing_blocks = self.get_missing_blocks(self.belief)
        if current_block in missing_blocks:
            return False
        return True

    def is_block_free(self, current_block):
        if self.is_block_on_board(current_block):
            for stack in self.belief.get_stacks():
                if current_block in stack.get_blocks():
                    if stack.get_above(current_block) is None:
                        return True
        return False

    def get_above_block(self, current_block):
        for stack in self.belief.get_stacks():
            if current_block in stack.get_blocks():
                return stack.get_above(current_block)
        return Block("Error")

    def get_below_block(self, current_block):
        for stack in self.belief.get_stacks():
            if current_block in stack.get_blocks():
                return stack.get_below(current_block)
        return Block("Error")

    def unstack_vs_pickup(self, curr_block):
        for stack in self.belief.get_stacks():
            if curr_block in stack.get_blocks():
                if stack.get_below(curr_block) is None:
                    return PickUp
        return Unstack

    def stack_vs_putdown(self, target_block):
        if target_block is None:
            return PutDown
        return Stack




    def revise_beliefs(self, perception: BlocksWorldPerception):
        """
        TODO: revise internal agent structured depending on whether what the agent *expects* to be true
        corresponds to what the agent perceives from the environment.
        :param perceived_world_state: the world state perceived by the agent
        :param previous_action_succeeded: whether the previous action succeeded or not
        """
        #raise NotImplementedError("not implemented yet; todo by student")

        self.belief = perception.current_world
        if not self.current_desire:
            self.current_desire = self.init_desires()

        if not self.belief:
            self.belief = perception.current_world



        print("Placed blocks:")
        print(self.get_placed_blocks(perception.current_world))

        print("Missing blocks:")
        print(self.get_missing_blocks(perception.current_world))

        print("Current intention:")
        print(self.current_intention)

        print("Current desire:")
        print(self.current_desire)

        print("Current belief:")
        print(self.belief)

        if perception.holding_block is not None:
            self.current_intention = [PutDown(Block(perception.holding_block))]



    def plan(self) -> Tuple[List[Block], List[BlocksWorldAction]]:
        # TODO: return the current desire from the set of possible / still required ones, on which the agent wants to focus next,
        # and the partial plan, as a sequence of `BlocksWorldAction' instances, that the agent wants to execute to achieve the current desire.
        intention = []

        current_block = copy(self.current_desire[0])

        print(f"Current block: {current_block}")

        # If the block is not on the table, save it for later
        if not self.is_block_on_board(current_block):
            self.current_desire.append(current_block)
            self.current_desire.pop(0)
            return self.current_desire, []

        # Block is in position, therefore lock it
        if self.is_block_in_position(current_block):
            self.current_desire.pop(0)
            return self.current_desire, [Lock(current_block)]

        # Block is not free, we have to look at the one above it
        if not self.is_block_free(current_block):
            block_above = self.get_above_block(current_block)
            self.current_desire.insert(0, block_above)
            return self.current_desire, [NoAction()]

        # If the block is free, we check if the target is available and in position and if so, we move it there, otherwise, dump it on the table for later
        if self.is_target_block_free(current_block) and self.is_block_in_position(self.get_target_block(current_block)):

            # Sanity check for unstack vs pickup and stack vs putdown
            if self.unstack_vs_pickup(current_block) == PickUp:
                intention.append(PickUp(current_block))
            else:
                intention.append(Unstack(current_block, self.get_below_block(current_block)))

            if self.stack_vs_putdown(current_block) == PutDown:
                intention.append(PutDown(current_block))
            else:
                intention.append(Stack(current_block, self.get_target_block(current_block)))
            self.current_desire.pop(0)
            return self.current_desire, intention

        else:
            if self.unstack_vs_pickup(current_block) == PickUp:
                intention.append(PickUp(current_block))
            else:
                intention.append(Unstack(current_block, self.get_below_block(current_block)))

            intention.append(PutDown(current_block))
            self.current_desire.append(current_block)
            self.current_desire.pop(0)
            return self.current_desire, intention


        return self.current_desire, intention


    def status_string(self):
        # TODO: return information about the agent's current state and current plan.
        return str(self) + " : PLAN MISSING"



class Tester(object):
    STEP_DELAY = 0.5
    TEST_SUITE = "tests/0e-large/"

    EXT = ".txt"
    SI  = "si"
    SF  = "sf"

    DYNAMICS_PROB = .5

    AGENT_NAME = "*A"

    def __init__(self):
        self._environment = None
        self._agents = []

        self._initialize_environment(Tester.TEST_SUITE)
        self._initialize_agents(Tester.TEST_SUITE)



    def _initialize_environment(self, test_suite: str) -> None:
        filename = test_suite + Tester.SI + Tester.EXT

        with open(filename) as input_stream:
            self._environment = DynamicEnvironment(BlocksWorld(input_stream=input_stream))


    def _initialize_agents(self, test_suite: str) -> None:
        filename = test_suite + Tester.SF + Tester.EXT

        agent_states = {}

        with open(filename) as input_stream:
            desires = BlocksWorld(input_stream=input_stream)
            agent = MyAgent(Tester.AGENT_NAME, desires)

            agent_states[agent] = desires
            self._agents.append(agent)

            self._environment.add_agent(agent, desires, None)

            print("Agent %s desires:" % str(agent))
            print(str(desires))


    def make_steps(self):
        print("\n\n================================================= INITIAL STATE:")
        print(str(self._environment))
        print("\n\n=================================================")

        completed = False
        nr_steps = 0

        while not completed:
            completed = self._environment.step()

            time.sleep(Tester.STEP_DELAY)
            print(str(self._environment))

            for ag in self._agents:
                print(ag.status_string())

            nr_steps += 1

            print("\n\n================================================= STEP %i completed." % nr_steps)

        print("\n\n================================================= ALL STEPS COMPLETED")





if __name__ == "__main__":
    tester = Tester()
    tester.make_steps()