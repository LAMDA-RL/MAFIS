#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "remote.h"
#include "boardgame.h"
#include "constant.h"

#include <iostream>
#include <string>


namespace py = pybind11;
class Miner {
    public:
        boardgame* game = new boardgame();
        Miner(py::object data) {
			game->my_player_id = data.attr("playerId").cast<int>();
			game->w = data.attr("gameinfo").attr("width").cast<int>();
			game->h = data.attr("gameinfo").attr("height").cast<int>();
			game->T = data.attr("gameinfo").attr("steps").cast<int>();

			int x = data.attr("posy").cast<int>();
			int y = data.attr("posx").cast<int>();
			game->maxE = data.attr("energy").cast<int>();

			game->player_list.push_back(player(game->my_player_id, x, y, game->maxE));

			game->trap_map.resize(game->h, std::vector<int>(game->w));

			//init data for trap
			py::list array_obstacles = data.attr("gameinfo").attr("obstacles").cast<py::list>();
			for (auto item: array_obstacles)
			{
				int x = item.attr("posy").cast<int>();
				int y = item.attr("posx").cast<int>();
				int type = item.attr("type").cast<int>();
				game->trap_map[x][y] = type;
			}

			// init data for gold
			py::list array_gold = data.attr("gameinfo").attr("golds").cast<py::list>();
			for (auto item: array_gold)
			{
				int x = item.attr("posy").cast<int>();
				int y = item.attr("posx").cast<int>();
				int amount = item.attr("amount").cast<int>();
				game->golds_map[{x, y}] = amount;
				game->trap_map[x][y] = O_GOLD;
			}

			game->free_turn = 0;

        }
		void update(py::object data) {
			py::list array_player = data.attr("players").cast<py::list>();
			game->player_list.clear();
			for (auto item: array_player)
			{
				int id = item.attr("playerId").cast<int>();
				int x = item.attr("posy").cast<int>();
				int y = item.attr("posx").cast<int>();
				int score = item.attr("score").cast<int>();
				int E = item.attr("energy").cast<int>();
				int state = item.attr("status").cast<int>();

				player p(id, x, y, E);
				p.score = score;
				p.state = state;
				game->player_list.push_back(p);
			}
			//init data for gold
			py::list array_gold = data.attr("golds").cast<py::list>();
			std::map<std::pair<int, int>, int> new_golds_map;
			for (auto item: array_gold)
			{
				int x = item.attr("posy").cast<int>();
				int y = item.attr("posx").cast<int>();
				int amount = item.attr("amount").cast<int>();
				new_golds_map[{x, y}] = amount;
			}
			// remove all cleared gold mine
			for (auto it : game->golds_map)
			{
				std::pair<int, int> key = it.first;
				if (new_golds_map.find(key) == new_golds_map.end())
				{
					game->trap_map[key.first][key.second] = O_LAND;
				}
			}
			game->golds_map = new_golds_map;
			// update data
			game->T--;
		}
		int get_best_move(int player_id) {
			game->my_player_id = player_id;
			int action = game->get_best_move(game->my_player_id);
			if (action < 0) {
				action = A_FREE;
			}
			return action;
		}
		void exit_game() {
			delete game;
			game = NULL;
		}
};
PYBIND11_MODULE(miner_cpp, m) {
    m.doc() = R"pbdoc(Pybind11 example plugin)pbdoc";
    py::class_<Miner>(m, "Miner")
        .def(py::init<py::object &>())
        .def("get_best_move", &Miner::get_best_move)
        .def("update", &Miner::update)
        .def("exit_game", &Miner::exit_game)
        ;
}
