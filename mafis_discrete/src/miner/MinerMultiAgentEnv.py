import json
import os
import math
from copy import deepcopy
from random import randrange
from typing import List

TreeID = 1
TrapID = 2
SwampID = 3

class ObstacleInfo:
    # initial energy for obstacles: Land (key = 0): -1, Forest(key = -1): 0 (random), Trap(key = -2): -10, Swamp (key = -3): -5
    types = {0: -1, -1: 0, -2: -10, -3: -5}

    def __init__(self):
        self.type = 0
        self.posx = 0
        self.posy = 0
        self.value = 0


class GoldInfo:
    def __init__(self):
        self.posx = 0
        self.posy = 0
        self.amount = 0

    def loads(self, data):
        golds = []
        for gd in data:
            g = GoldInfo()
            g.posx = gd["posx"]
            g.posy = gd["posy"]
            g.amount = gd["amount"]
            golds.append(g)
        return golds


class PlayerInfo:
    STATUS_PLAYING = 0
    STATUS_ELIMINATED_WENT_OUT_MAP = 1
    STATUS_ELIMINATED_OUT_OF_ENERGY = 2
    STATUS_ELIMINATED_INVALID_ACTION = 3
    STATUS_STOP_EMPTY_GOLD = 4
    STATUS_STOP_END_STEP = 5

    def __init__(self, id):
        self.playerId = id
        self.score = 0
        self.energy = 0
        self.posx = 0
        self.posy = 0
        self.lastAction = -1
        self.status = PlayerInfo.STATUS_PLAYING
        self.freeCount = 0


class GameInfo:
    def __init__(self):
        self.numberOfPlayers = 1
        self.width = 0
        self.height = 0
        self.steps = 100
        self.golds: List[GoldInfo] = []
        self.obstacles: List[ObstacleInfo] = []

    def loads(self, data):
        m = GameInfo()
        m.width = data["width"]
        m.height = data["height"]
        m.golds = GoldInfo().loads(data["golds"])
        m.obstacles = data["obstacles"]
        m.numberOfPlayers = data["numberOfPlayers"]
        m.steps = data["steps"]
        return m


class UserMatch:
    def __init__(self):
        self.playerId = 1
        self.posx = 0
        self.posy = 0
        self.energy = 50
        self.gameinfo = GameInfo()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        # return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class StepState:
    def __init__(self):
        self.players = []
        self.golds: List[GoldInfo] = []
        self.changedObstacles = []

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class GameSocket:
    bog_energy_chain = {-5: -20, -20: -40, -40: -100, -100: -100}

    def __init__(self, n_agents=4):
        self.n_agents = n_agents
        self.stepCount = 0
        self.maxStep = 0
        self.mapdir = "miner/Maps"  # where to load all pre-defined maps
        self.mapid = ""
        self.userMatch = UserMatch()
        # self.user = PlayerInfo(1)
        self.stepState = StepState()
        self.maps = {}  # key: map file name, value: file content
        self.map = []  # running map info: 0->Land, -1->Forest, -2->Trap, -3:Swamp, >0:Gold
        self.energyOnMap = []  # self.energyOnMap[x][y]: <0, amount of energy which player will consume if it move into (x,y)
        self.E = 50
        self.resetFlag = True
        self.craftUsers = []  # players that craft at current step - for calculating amount of gold
        self.bots = []
        self.craftMap = {}  # cells that players craft at current step, key: x_y, value: number of players that craft at (x,y)

    def init_bots(self):
        self.bots = [PlayerInfo(id) for id in range(self.n_agents)]  # use bot1(id=2), bot2(id=3), bot3(id=4)
        for bot in self.bots:  # at the beginning, all bots will have same position, energy as player
            bot.posx = self.userMatch.posx
            bot.posy = self.userMatch.posy
            bot.energy = self.userMatch.energy
            bot.lastAction = -1
            bot.status = PlayerInfo.STATUS_PLAYING
            bot.score = 0
            self.stepState.players.append(bot)
        self.userMatch.gameinfo.numberOfPlayers = len(self.stepState.players)
        # print("numberOfPlayers: ", self.userMatch.gameinfo.numberOfPlayers)

    def check_energy(self, energy, posx, posy):
        if posx < 0 or posx >= self.userMatch.gameinfo.width:
            return False
        if posy < 0 or posy >= self.userMatch.gameinfo.height:
            return False
        if self.map[posy][posx] == -1:
            energy -= 20
        elif self.map[posy][posx] == 0:
            energy += self.energyOnMap[posy][posx]
        elif self.map[posy][posx] == -2:
            energy += self.energyOnMap[posy][posx]
        elif self.map[posy][posx] == -3:
            energy += self.energyOnMap[posy][posx]
        else:
            energy -= 4
        return energy > 0

    def check_gold(self, posx, posy):
        amount = self.map[posy][posx]
        if amount < 50:
            return False
        count = {}
        for bot in self.bots:
            dist = abs(bot.posx-posx) + abs(bot.posy-posy)
            count[dist] = count.get(dist, 0) + 1
        return 50 * (count.get(1, 0) + 2 * count.get(0, 0)) <= amount

    def get_avails(self, user: PlayerInfo, golds: List[GoldInfo]=[]):
        assert len(self.stepState.golds) > 0 and any(gold.amount > 0 for gold in self.stepState.golds)
        
        if len(golds) == 0:
            for gold in self.stepState.golds:
                if gold.amount <= 0:
                    continue
                dist = abs(gold.posx - user.posx) + abs(gold.posy - user.posy)
                if dist < self.maxStep - self.stepCount:
                    golds.append(gold)
        if len(golds) == 0:
            golds = self.stepState.golds

        gold_x = [gold.posx for gold in golds]
        gold_y = [gold.posy for gold in golds]
        min_x, max_x = min(gold_x), max(gold_x)
        min_y, max_y = min(gold_y), max(gold_y)

        avails = [1] * 6
        if not self.check_energy(user.energy, user.posx-1, user.posy) or user.posx-1 < min_x:
            avails[0] = 0
        if not self.check_energy(user.energy, user.posx+1, user.posy) or user.posx+1 > max_x:
            avails[1] = 0
        if not self.check_energy(user.energy, user.posx, user.posy-1) or user.posy-1 < min_y:
            avails[2] = 0
        if not self.check_energy(user.energy, user.posx, user.posy+1) or user.posy+1 > max_y:
            avails[3] = 0
        
        force_actions = []
        if avails[0] == 1 and self.check_gold(user.posx-1, user.posy):
            force_actions.append(0)
        if avails[1] == 1 and self.check_gold(user.posx+1, user.posy):
            force_actions.append(1)
        if avails[2] == 1 and self.check_gold(user.posx, user.posy-1):
            force_actions.append(2)
        if avails[3] == 1 and self.check_gold(user.posx, user.posy+1):
            force_actions.append(3)
        if len(force_actions) > 0:
            for i in range(4):
                avails[i] = 0
            for i in force_actions:
                avails[i] = 1

        if user.energy >= self.E - int(self.E / 4):
            avails[4] = 0
        
        avails[5] = 0
        if self.map[user.posy][user.posx] > 0:
            if user.energy - 5 > 0:
                avails[5] = 1
            count = 0
            for bot in self.bots:
                if bot.posx == user.posx and bot.posy == user.posy:
                    count += 1
            if 50*count <= self.map[user.posy][user.posx] or len(golds) == 1:
                for i in range(5):
                    avails[i] = 0
        
        if all(avail==0 for avail in avails):
            avails = [0] * 6
            avails[4] = 1

        # print(user.playerId, avails)
        return avails
    
    def reset(self, requests):  # load new game by given request: [map id (filename), posx, posy, initial energy]
        # load new map
        self.reset_map(requests[0])
        self.userMatch.posx = int(requests[1])
        self.userMatch.posy = int(requests[2])
        self.userMatch.energy = int(requests[3])
        self.userMatch.gameinfo.steps = int(requests[4])
        self.maxStep = self.userMatch.gameinfo.steps

        # init data for players
        # self.user.posx = self.userMatch.posx  # in
        # self.user.posy = self.userMatch.posy
        # self.user.energy = self.userMatch.energy
        # self.user.status = PlayerInfo.STATUS_PLAYING
        # self.user.score = 0
        # self.stepState.players = [self.user]
        self.stepState.players = []
        self.E = self.userMatch.energy
        self.resetFlag = True
        self.init_bots()
        self.stepCount = 0

        # data = self.userMatch.to_json()
        # print("data:", data)
        # exit()

    def reset_map(self, id):  # load map info
        self.mapId = id
        self.map = deepcopy(self.maps[self.mapId])
        self.userMatch = self.map_info(self.map)
        #print(self.map)
        self.stepState.golds = self.userMatch.gameinfo.golds
        #self.map = deepcopy(self.maps[self.mapId])
        self.energyOnMap = deepcopy(self.maps[self.mapId])
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                if self.map[x][y] > 0:  # gold
                    self.energyOnMap[x][y] = -4
                else:  # obstacles
                    self.energyOnMap[x][y] = ObstacleInfo.types[self.map[x][y]]

    def connect(self):  # simulate player's connect request
        # print("Connected to server.")
        # load all pre-defined maps from mapDir
        for filename in os.listdir(self.mapdir):
            # print("Found: " + filename)
            with open(os.path.join(self.mapdir, filename), 'r') as f:
                self.maps[filename] = json.loads(f.read())

    def map_info(self, map):  # get map info
        # print(map)
        userMatch = UserMatch()
        userMatch.gameinfo.height = len(map)
        userMatch.gameinfo.width = len(map[0])
        i = 0
        max_gold_num = 10000
        while i < len(map):
            j = 0
            while j < len(map[i]):
                if map[i][j] > 0:  # gold
                    g = GoldInfo()
                    g.posx = j
                    g.posy = i
                    g.amount = (randrange(min(25, max(int(max_gold_num/50), 1))) + 1) * 50
                    self.map[i][j] = g.amount
                    max_gold_num -= g.amount
                    userMatch.gameinfo.golds.append(g)
                else:  # obstacles
                    o = ObstacleInfo()
                    o.posx = j
                    o.posy = i
                    o.type = -map[i][j]
                    o.value = ObstacleInfo.types[map[i][j]]
                    userMatch.gameinfo.obstacles.append(o)
                j += 1
            i += 1
        #print("max_gold_num=",max_gold_num)
        return userMatch

    def receive(self):  # send data to player (simulate player's receive request)
        if self.resetFlag:  # for the first time -> send game info
            self.resetFlag = False
            data = self.userMatch.to_json()
            for (bot) in self.bots:
                bot.new_game(data)
            #print(data)
            return data
        else:  # send step state
            self.stepCount = self.stepCount + 1
            if self.stepCount >= self.maxStep:
                for player in self.stepState.players:
                    player.status = PlayerInfo.STATUS_STOP_END_STEP
            data = self.stepState.to_json()
            for (bot) in self.bots:  # update bots' state
                bot.new_state(data)
            # print(data)
            return data

    def step(self, actions):
        assert len(actions) == len(self.bots)
        self.resetFlag = False
        self.stepState.changedObstacles = []
        # print("Action = ", action)
        self.craftUsers = []
        # for user, action in zip(self.users, actions):
        # self.step_action(user, action)
        for bot, action in zip(self.bots, actions):
            if bot.status == PlayerInfo.STATUS_PLAYING:
                # action = bot.next_action()
                bot.lastAction = action
                # print("Bot Action: ", action)
                # print("bot:", vars(bot))
                self.step_action(bot, action)
        self.action_5_craft()
        for c in self.stepState.changedObstacles:
            self.map[c["posy"]][c["posx"]] = -c["type"]
            self.energyOnMap[c["posy"]][c["posx"]] = c["value"]

        # print(self.stepState.changedObstacles)
        # for user in self.bots:
        #     if user.status != 0:
        #         print("error:", vars(user))
        #         raise

    def send(self, message):  # receive message from player (simulate send request from player)
        if message.isnumeric():  # player send action
            self.resetFlag = False
            self.stepState.changedObstacles = []
            action = int(message)
            # print("Action = ", action)
            self.user.lastAction = action
            self.craftUsers = []
            self.step_action(self.user, action)
            for bot in self.bots:
                if bot.info.status == PlayerInfo.STATUS_PLAYING:
                    action = bot.next_action()
                    bot.info.lastAction = action
                    # print("Bot Action: ", action)
                    self.step_action(bot.info, action)
            self.action_5_craft()
            for c in self.stepState.changedObstacles:
                self.map[c["posy"]][c["posx"]] = -c["type"]
                self.energyOnMap[c["posy"]][c["posx"]] = c["value"]

        else:  # reset game
            requests = message.split(",")
            print("Reset game: ", requests)
            self.reset(requests)

    def step_action(self, user, action):
        switcher = {
            0: self.action_0_left,
            1: self.action_1_right,
            2: self.action_2_up,
            3: self.action_3_down,
            4: self.action_4_free,
            5: self.action_5_craft_pre
        }
        func = switcher.get(action, self.invalidAction)
        func(user)

    def action_5_craft_pre(self, user):  # collect players who craft at current step
        user.freeCount = 0
        if self.map[user.posy][user.posx] <= 0:  # craft at the non-gold cell
            user.energy -= 10
            if user.energy <= 0:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated
        else:
            user.energy -= 5
            if user.energy > 0:
                self.craftUsers.append(user)
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    count = self.craftMap[key]
                    self.craftMap[key] = count + 1
                else:
                    self.craftMap[key] = 1
            else:
                user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
                user.lastAction = 6 #eliminated

    def action_0_left(self, user):  # user go left
        user.freeCount = 0
        user.posx = user.posx - 1
        if user.posx < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_1_right(self, user):  # user go right
        user.freeCount = 0
        user.posx = user.posx + 1
        if user.posx >= self.userMatch.gameinfo.width:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_2_up(self, user):  # user go up
        user.freeCount = 0
        user.posy = user.posy - 1
        if user.posy < 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_3_down(self, user):  # user go right
        user.freeCount = 0
        user.posy = user.posy + 1
        if user.posy >= self.userMatch.gameinfo.height:
            user.status = PlayerInfo.STATUS_ELIMINATED_WENT_OUT_MAP
            user.lastAction = 6 #eliminated
        else:
            self.go_to_pos(user)

    def action_4_free(self, user):  # user free
        user.freeCount += 1
        if user.freeCount == 1:
            user.energy += int(self.E / 4)
        elif user.freeCount == 2:
            user.energy += int(self.E / 3)
        elif user.freeCount == 3:
            user.energy += int(self.E / 2)
        else:
            user.energy = self.E
        if user.energy > self.E:
            user.energy = self.E

    def action_5_craft(self):
        craftCount = len(self.craftUsers)
        # print ("craftCount",craftCount)
        if (craftCount > 0):
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                c = self.craftMap[key]
                m = min(math.ceil(self.map[y][x] / c), 50)
                user.score += m
                # print ("user", user.playerId, m)
            for user in self.craftUsers:
                x = user.posx
                y = user.posy
                key = str(user.posx) + "_" + str(user.posy)
                if key in self.craftMap:
                    c = self.craftMap[key]
                    del self.craftMap[key]
                    m = min(math.ceil(self.map[y][x] / c), 50)
                    self.map[y][x] -= m * c
                    if self.map[y][x] < 0:
                        self.map[y][x] = 0
                        self.energyOnMap[y][x] = ObstacleInfo.types[0]
                    for g in self.stepState.golds:
                        if g.posx == x and g.posy == y:
                            g.amount = self.map[y][x]
                            if g.amount == 0:
                                self.stepState.golds.remove(g)
                                self.add_changed_obstacle(x, y, 0, ObstacleInfo.types[0])
                                if len(self.stepState.golds) == 0:
                                    for player in self.stepState.players:
                                        player.status = PlayerInfo.STATUS_STOP_EMPTY_GOLD
                            break
            self.craftMap = {}

    def invalidAction(self, user):
        user.status = PlayerInfo.STATUS_ELIMINATED_INVALID_ACTION
        user.lastAction = 6 #eliminated

    def go_to_pos(self, user):  # player move to cell(x,y)
        if self.map[user.posy][user.posx] == -1:
            user.energy -= randrange(16) + 5
        elif self.map[user.posy][user.posx] == 0:
            user.energy += self.energyOnMap[user.posy][user.posx]
        elif self.map[user.posy][user.posx] == -2:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 0, ObstacleInfo.types[0])
        elif self.map[user.posy][user.posx] == -3:
            user.energy += self.energyOnMap[user.posy][user.posx]
            self.add_changed_obstacle(user.posx, user.posy, 3,
                                      self.bog_energy_chain[self.energyOnMap[user.posy][user.posx]])
        else:
            user.energy -= 4
        if user.energy <= 0:
            user.status = PlayerInfo.STATUS_ELIMINATED_OUT_OF_ENERGY
            user.lastAction = 6 #eliminated

    def add_changed_obstacle(self, x, y, t, v):
        added = False
        for o in self.stepState.changedObstacles:
            if o["posx"] == x and o["posy"] == y:
                added = True
                break
        if added == False:
            o = {}
            o["posx"] = x
            o["posy"] = y
            o["type"] = t
            o["value"] = v
            self.stepState.changedObstacles.append(o)

    def close(self):
        print("Close socket.")
