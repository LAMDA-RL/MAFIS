#include "player.h"
#include <iostream>

player::player(int id, int x, int y, int E)
{
    this->id = id;
    this->x = x;
    this->y = y;
    this->E = E;
    this->state = -1;
    this->score = 0;
    this->rest_count = 0;
}

void player::show()
{
    std::cout << "PlayerId: " << this->id << std::endl;
    std::cout << "posx: " << this->x << std::endl;
    std::cout << "posy: " << this->y << std::endl;
    std::cout << "E: " << this->E << std::endl;
    std::cout << "state: " << this->state << std::endl;
    std::cout << "restc: " << this->rest_count << std::endl << std::endl;
}
