#ifndef BOARDGAME_H
#define BOARDGAME_H
#include "player.h"
#include "gold.h"
#include "bfs.h"

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <queue>

class boardgame
{
public:
    std::vector<player> player_list;
    std::vector<std::vector<int>> trap_map;
    std::map<std::pair<int,int>, int> golds_map;
    int free_turn = 0;
    int w, h, T, my_player_id, maxE;
    boardgame();
    boardgame(std::string raw_json);
    void update(std::string raw_json);

    int manhattan_distance(int x1, int y1, int x2, int y2);
    
    std::vector<gold> golds_list(int player_id);
    player get_player(int player_id);

    int can_move(int player_id, int direction);
    int can_craft(int player_id);
    int object_at(int x, int y);

    int count_turn_from_path(int player_id, std::vector<int> trap_path);
    int get_direction(int x1, int y1, int x2, int y2);
    
    // E, trap_path, direction
    std::tuple<int, std::vector<int>,  int> best_direction(int from_x, int from_y, int to_x, int to_y) ;
    
    bool can_get_more_gold(int player_id);
    int count_player_at(int x, int y);
    int count_danger_player_at(int x, int y, int range);

    gold best_mine(int player_id);

    // god bless us <3
    int get_best_move(int player_id);

    void show();
};

#endif
