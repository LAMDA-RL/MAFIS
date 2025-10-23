import random
import numpy as np
from miner_cpp import Miner
from miner.MinerMultiAgentEnv import GameSocket, SwampID, TrapID, TreeID


ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5


class cell:
    step = 10000
    energy = 0
    trace = -1


class BlackPantherBot:

    def __init__(self, env: GameSocket):
        self._env = env
        self._pre_state = {}

    def get_obstacle(self, x, y):  # Getting the kind of the obstacle at cell(x,y)
        for cell in self._env.userMatch.gameinfo.obstacles:
            if x == cell.posx and y == cell.posy:
                return cell.type
        return -1  # No obstacle at the cell (x,y)
    
    def gold_amount(self, x, y):
        for cell in self._env.userMatch.gameinfo.golds:
            if x == cell.posx and y == cell.posy:
                return cell.amount
        return 0
    
    def get_state(self, player_id=0):
        pre_state = self._pre_state.get(player_id, [])
        mode = 0 if len(pre_state) == 0 else 1
        user = self._env.bots[player_id]
        # Building the map
        max_x = self._env.userMatch.gameinfo.width - 1
        max_y = self._env.userMatch.gameinfo.height - 1

        view = np.zeros([max_y + 1, max_x + 1], dtype=int)
        if mode == 1:
            aftermap = (max_y + 1) * (max_x + 1)
            pos_x = pre_state[aftermap + 0]
            pos_y = pre_state[aftermap + 1]
            energy = pre_state[aftermap + 2]
            freeCount = pre_state[aftermap + 3]
            stepLeft = pre_state[aftermap + 4]
            aftermap += 5
            players_x = []
            players_y = []
            change_x = []
            change_y = []
            cnt = 0
            for player in self._env.bots:
                if player.playerId != user.playerId:
                    players_x.append(pre_state[aftermap + 0])
                    players_y.append(pre_state[aftermap + 1])
                    aftermap += 2
                    if players_x[cnt] != player.posx or players_y[cnt] != player.posy:
                        change_x.append(player.posx)
                        change_y.append(player.posy)
                    cnt += 1
            if pos_x != user.posx or pos_y != user.posy:
                change_x.append(user.posx)
                change_y.append(user.posy)            

        for i in range(max_x + 1):
            for j in range(max_y + 1):
                obs_type = self.get_obstacle(i, j)
                if obs_type == 0:  # land
                    view[j, i] = 0
                if obs_type == TreeID:  # Tree
                    view[j, i] = -TreeID
                if obs_type == TrapID:  # Trap
                    view[j, i] = -10
                if obs_type == SwampID:  # Swamp
                    if mode == 0:
                        view[j, i] = -5
                    elif mode == 1:
                        view[j, i] = pre_state[j * (max_x + 1) + i]
                        flag = False
                        for d in range(len(change_x)):
                            if change_x[d] == i and change_y[d] == j:
                                flag = True
                        if flag == True:
                            if view[j, i] == -5:
                                view[j, i] = -20
                            elif view[j, i] == -20:
                                view[j, i] = -40
                            elif view[j, i] == -40:
                                view[j, i] = -100
                amount = self.gold_amount(i, j)
                if amount > 0:
                    view[j, i] = amount

        DQNState = view.flatten().tolist()#Flattening the map matrix to a vector
        
        # Add position and energy of agent to the DQNState
        DQNState.append(user.posx)
        DQNState.append(user.posy)
        DQNState.append(user.energy)
        if mode == 0:
            DQNState.append(0)
            DQNState.append(100)
        else:
            if energy >= user.energy:
                DQNState.append(0)
            else:
                DQNState.append(freeCount + 1)
            DQNState.append(stepLeft - 1)            
                        
        #Add position of bots 
        for player in self._env.bots:
            if player.playerId != user.playerId:
                DQNState.append(player.posx)
                DQNState.append(player.posy)
                
        #Convert the DQNState from list to array
        DQNState = np.array(DQNState)

        self._pre_state[player_id] = DQNState

        return DQNState

    def change_cell(self, Cell):
        if Cell == -10:
            return 0
        if Cell == 0:
            return 0
        if Cell == -5:
            return -20
        if Cell == -20:
            return -40
        if Cell == -40:
            return -100
        return Cell
    
    def player_target(self, map, pos_x, pos_y, energy, stepLeft):
        max_x = 21
        max_y = 9
        bfs = []
        for y in range(max_y):
            bfsrow = []
            for x in range(max_x):
                bfscell = cell()
                if map[y][x] <= -50:
                    bfscell.step = -3
                bfsrow.append(bfscell)
            bfs.append(bfsrow)
        bfs[pos_y][pos_x].step = 0
        bfs[pos_y][pos_x].energy = energy
        bfs[pos_y][pos_x].trace = 5

        dx = [-1, 1, 0, 0, 0, 0]
        dy = [ 0, 0,-1, 1, 0, 0]
        increase = [0, 12, 28, 50, 100]
        target_x = 0
        target_y = 0
        target_gold = -10000
        for stp in range(int(stepLeft + 1)):
            if target_gold > 0:
                return stp, target_x, target_y
            for y in range(max_y):
                for x in range(max_x):
                    if bfs[y][x].step == stp:
                        if target_gold < map[y][x] and map[y][x] > 0:
                            target_x = x
                            target_y = y
                            target_gold = map[y][x]
                        for i in range(4):
                            nx = x + dx[i]
                            ny = y + dy[i]
                            if nx < 0 or nx >= max_x or ny < 0 or ny >= max_y:
                                continue
                            if bfs[ny][nx].step < 0:
                                continue
                            necessary = -map[ny][nx]
                            lost = -map[ny][nx]
                            if map[ny][nx] > 0:
                                necessary = 4
                                lost = 4
                            elif map[ny][nx] == 0:
                                necessary = 1
                                lost = 1
                            elif map[ny][nx] == -1:
                                necessary = 20
                                lost = 13
                            ft = 0
                            while bfs[y][x].energy + increase[ft] <= necessary:
                                ft += 1
                            newcell = cell()
                            newcell.step = bfs[y][x].step + 1 + ft
                            newcell.energy = bfs[y][x].energy - lost + increase[ft]
                            newcell.trace = i
                            if bfs[ny][nx].step > newcell.step:
                                bfs[ny][nx] = newcell
                            elif bfs[ny][nx].step == newcell.step and bfs[ny][nx].energy < newcell.energy:
                                bfs[ny][nx] = newcell
        return -1, -1, -1

    def find_gold(self, map, pos_x, pos_y, energy, stepLeft, players_x, players_y, players_E, mod):
        max_x = 21
        max_y = 9
        bfs = []
        for y in range(max_y):
            bfsrow = []
            for x in range(max_x):
                bfscell = cell()
                if map[y][x] <= -50:
                    bfscell.step = -3
                bfsrow.append(bfscell)
            bfs.append(bfsrow)
        bfs[pos_y][pos_x].step = 0
        bfs[pos_y][pos_x].energy = energy
        bfs[pos_y][pos_x].trace = 5

        dx = [-1, 1, 0, 0, 0, 0]
        dy = [ 0, 0,-1, 1, 0, 0]
        increase = [0, 12, 28, 50, 100]
        for stp in range(int(stepLeft + 1)):
            for y in range(max_y):
                for x in range(max_x):
                    if bfs[y][x].step == stp:
                        for i in range(4):
                            nx = x + dx[i]
                            ny = y + dy[i]
                            if nx < 0 or nx >= max_x or ny < 0 or ny >= max_y:
                                continue
                            if bfs[ny][nx].step < 0:
                                continue
                            necessary = -map[ny][nx]
                            lost = -map[ny][nx]
                            if map[ny][nx] > 0:
                                necessary = 4
                                lost = 4
                            elif map[ny][nx] == 0:
                                necessary = 1
                                lost = 1
                            elif map[ny][nx] == -1:
                                necessary = 20
                                lost = 13
                            ft = self.count_free_time(bfs[y][x].energy, 50, 0, necessary)
                            newcell = cell()
                            newcell.step = bfs[y][x].step + 1 + ft
                            newcell.energy = bfs[y][x].energy - lost + increase[ft]
                            newcell.trace = i
                            if bfs[ny][nx].step > newcell.step:
                                bfs[ny][nx] = newcell
                            elif bfs[ny][nx].step == newcell.step and bfs[ny][nx].energy < newcell.energy:
                                bfs[ny][nx] = newcell
        
        if mod == 1:
            playercnt = []
            playersum = []
            for y in range(max_y):
                row = []
                Row = []
                for x in range(max_x):
                    row.append(0)
                    Row.append(0)
                playercnt.append(row)
                playersum.append(Row)
            #print(players_E)
            for i in range(len(players_E)):
                if players_x[i] < 0 or players_x[i] >= max_x or players_y[i] < 0 or players_y[i] >= max_y:
                    continue
                sstep, sx, sy = self.player_target(map, players_x[i], players_y[i], players_E[i], stepLeft)
                if sstep > 0:
                    playercnt[sy][sx] += 1
                    playersum[sy][sx] += sstep

        target_x = 0
        target_y = 0
        target_step = 1
        target_gold = -10000
        for y in range(max_y):
            for x in range(max_x):
                if map[y][x] > 0:
                    if mod == 1:
                        GOLD = (map[y][x] + 50 * playersum[y][x] - 50 * playercnt[y][x] * bfs[y][x].step) / (playercnt[y][x] + 1)
                        if GOLD > map[y][x]:
                            GOLD = map[y][x]
                        if GOLD < 0:
                            GOLD = 0
                    else:
                        GOLD = map[y][x]

                    step_go = bfs[y][x].step
                    if step_go >= stepLeft:
                        continue
                    gold_craft, cur_energy, step_craft = self.craft(GOLD, bfs[y][x].energy, stepLeft - step_go)
                    total_step = step_go + step_craft
                    if target_gold * total_step < gold_craft * target_step:
                        target_x = x
                        target_y = y
                        target_step = total_step
                        target_gold = gold_craft
                    if mod > 0:
                        change = []
                        action = bfs[y][x].trace
                        back_x = x
                        back_y = y
                        while bfs[back_y][back_x].trace < 4:
                            change.append(map[back_y][back_x])
                            map[back_y][back_x] = self.change_cell(map[back_y][back_x])
                            action = bfs[back_y][back_x].trace
                            back_x -= dx[action]
                            back_y -= dy[action]

                        na, ng, ns = self.find_gold(map, x, y, cur_energy, stepLeft - total_step,
                                            players_x, players_y, players_E, mod - 1)
                        gold_craft += ng
                        total_step += ns
                        if target_gold * total_step < gold_craft * target_step:
                            target_x = x
                            target_y = y
                            target_step = total_step
                            target_gold = gold_craft
                        
                        action = bfs[y][x].trace
                        back_x = x
                        back_y = y
                        dem = 0
                        while bfs[back_y][back_x].trace < 4:
                            map[back_y][back_x] = change[dem]
                            dem += 1
                            action = bfs[back_y][back_x].trace
                            back_x -= dx[action]
                            back_y -= dy[action]
        if target_gold < 0:
            return 4, 0, 0
        action = bfs[target_y][target_x].trace
        back_x = target_x
        back_y = target_y
        while bfs[back_y][back_x].trace < 4:
            action = bfs[back_y][back_x].trace
            back_x -= dx[action]
            back_y -= dy[action]
        return action, target_gold, target_step

    def craft(self, gold, energy, stepLeft):
        increase = [0, 12, 28, 50]
        lancuoi = gold % 50
        gold = gold // 50
        if lancuoi > 0:
            gold += 1
        else:
            lancuoi = 50
        step = min(gold, stepLeft)
        if step * 4 < energy:
            return step * 50 + 50 - lancuoi, energy - step * 4, step
        step = energy // 4
        gold -= step
        energy -= step * 4
        stepLeft -= step
        if stepLeft <= 4:
            ft = 1
        elif stepLeft <= 9:
            ft = 2
        else:
            ft = 3
        ng, ne, ns = self.craft(gold * 50 + 50 - lancuoi, min(energy + increase[ft], 50), stepLeft - ft)
        return step * 50 + ng + 50 - lancuoi, ne, step + ns

    def next_action(self, state):
        max_x = 21
        max_y = 9
        map = []
        for i in range(0, max_y):
            row = []
            for j in range(0, max_x):
                row.append(state[i * max_x + j])
            map.append(row)
        aftermap = max_x * max_y
        pos_x = state[aftermap + 0]
        pos_y = state[aftermap + 1]
        energy = state[aftermap + 2]
        freeCount = state[aftermap + 3]
        stepLeft = state[aftermap + 4]
        aftermap += 5
        players_x = []
        players_y = []
        players_E = []
        num_of_player = (len(state) - aftermap) // 2
        for i in range(num_of_player):
            players_x.append(state[aftermap + 0])
            players_y.append(state[aftermap + 1])
            players_E.append(max(stepLeft // 2, 1))
            aftermap += 2
        
        dx = [-1, 1, 0, 0, 0, 0]
        dy = [ 0, 0,-1, 1, 0, 0]
        #print(map)
        action, _, _ = self.find_gold(map, pos_x, pos_y, energy, stepLeft, players_x, players_y, players_E, 0)
        #print(action)
        #print(map)
        next_x = pos_x + dx[action]
        next_y = pos_y + dy[action]
        necessary = -map[next_y][next_x]
        if action == 5:
            necessary = 5
        elif map[next_y][next_x] > 0:
            necessary = 4
        elif map[next_y][next_x] == 0:
            necessary = 1
        elif map[next_y][next_x] == -1:
            necessary = 20

        maxE = stepLeft * 5 + 1
        playercnt = []
        playersum = []
        for y in range(max_y):
            row = []
            Row = []
            for x in range(max_x):
                row.append(0)
                Row.append(0)
            playercnt.append(row)
            playersum.append(Row)
        for i in range(len(players_E)):
            if players_x[i] < 0 or players_x[i] >= max_x or players_y[i] < 0 or players_y[i] >= max_y:
                continue
            sstep, sx, sy = self.player_target(map, players_x[i], players_y[i], players_E[i], stepLeft)
            if sstep > 0:
                playercnt[sy][sx] += 1
                playersum[sy][sx] += sstep

        if map[pos_y][pos_x] > 0 and playercnt[pos_y][pos_x] > 0:
            x = pos_x
            y = pos_y
            GOLD = (map[y][x] + 50 * playersum[y][x]) / (playercnt[y][x] + 1)
            if GOLD > map[y][x]:
                GOLD = map[y][x]
            if GOLD < 0:
                GOLD = 0
            lancuoi = GOLD % 50
            GOLD = GOLD // 50
            if lancuoi > 0:
                GOLD += 1
            else:
                lancuoi = 50
            maxE = min(maxE, GOLD * 5 + 1)

        if self.count_free_time(energy, maxE, freeCount, necessary) > 0:
            action = 4
        return action

    def count_free_time(self, energy, maxE, freeCount, necessary):
        old_energy = 0
        if freeCount == 1:
            old_energy = 12
        if freeCount == 2:
            old_energy = 28
        if freeCount == 3:
            old_energy = 50
        energy -= old_energy
        if maxE > 50:
            maxE = 50
        if maxE < necessary:
            maxE = necessary
        increase = [0, 12, 28, 50, 100]
        res = 0
        if energy < maxE - 24:
            res = 2
        if energy < maxE - 42:
            res = 3
        if freeCount == 0:
            if energy > necessary:
                return 0
        while energy + old_energy + increase[res] <= necessary:
            res += 1
        return res - freeCount

    def check(self, action, state):
        max_x = 21
        max_y = 9
        map = []
        for i in range(0, max_y):
            row=[]
            for j in range(0, max_x):
                row.append(state[j * max_y + i])
            map.append(row)
        aftermap = max_x * max_y
        pos_x = state[aftermap + 0]
        pos_y = state[aftermap + 1]
        pos_x = min(max_x - 1, pos_x)
        pos_x = min(0, pos_x)
        pos_y = max(max_y - 1, pos_y)
        pos_y = max(0, pos_y)
        energy = state[aftermap + 2]
        necessary = 0
        if action == 0:
            if pos_x == 0:
                return False
        if action == 1:
            if pos_x == max_x - 1:
                return False
        if action == 2:
            if pos_y == 0:
                return False
        if action == 3:
            if pos_y == max_y - 1:
                return False
        if action == 0 or action == 1 or action == 2 or action == 3:
            necessary = -map[pos_y][pos_x]
            if map[pos_y][pos_x] > 0:
                necessary = 4
            elif map[pos_y][pos_x] == 0:
                necessary = 1
            elif map[pos_y][pos_x] == -1:
                necessary = 20
            elif map[pos_y][pos_x] == -2:
                necessary = 10
            if energy <= necessary:
                return False
            return True
        if action == 5:
            if map[pos_y][pos_x] <= 0:
                return False
            necessary = 5
            if energy <= necessary:
                return False
            return True
        return True

    def get_action(self, player_id):
        state = self.get_state(player_id)
        action = self.next_action(state)
        return action


class DeepMindBot:

    def __init__(self, env: GameSocket):
        self._env = env
        self._pre_state = {}
        self.heuristic = Miner(self._env.userMatch)

    def get_action(self, player_id):
        self.heuristic.update(self._env.stepState)
        action = self.heuristic.get_best_move(player_id)
        avails = self._env.get_avails(self._env.bots[player_id])
        if avails[action] == 0:
            avail_ids = [k for k, x in enumerate(avails) if x==1]
            action = random.choice(avail_ids)
        return action


class GreedyBot:

    def __init__(self, env: GameSocket, topk=2):
        self._env = env
        self.topk = topk
        self.target_golds = {}

    def get_action(self, player_id):
        user = self._env.bots[player_id]
        if player_id not in self.target_golds or self.target_golds[player_id].amount <= 50:
            goals = sorted(self._env.stepState.golds, key=lambda gold: abs(gold.posx - user.posx) + abs(gold.posy - user.posy))
            goal = random.choice(goals[:self.topk])
            self.target_golds[player_id] = goal
        else:
            goal = self.target_golds[player_id]
        avails = self._env.get_avails(user, [goal])
        if avails[ACTION_CRAFT] == 1:
            return ACTION_CRAFT
        avail_ids = [k for k, x in enumerate(avails) if x==1]
        action = random.choice(avail_ids)
        return action
