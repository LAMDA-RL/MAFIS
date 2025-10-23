#include "gold.h"
#include <iostream>

gold::gold(int posx, int posy, int amount)
{
    this->x = posx;
    this->y = posy;
    this->amount = amount;
}

void gold::show()
{
    std::cout << "posx = " << this->x << std::endl;
    std::cout << "posy = " << this->y << std::endl;
    std::cout << "amount = " << this->amount << std::endl
              << std::endl;
}
