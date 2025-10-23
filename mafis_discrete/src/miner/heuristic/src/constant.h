#ifndef CONSTANT_H
#define CONSTANT_H
#include <string>

const int DAMAGE[5] = {1, 3, 2, 3, 4};
const int DIG_COST = 5;

const int O_LAND = 0;
const int O_FOREST = 1;
const int O_TRAP = 2;
const int O_SWAMP = 3;
const int O_GOLD = 4;

const int OK = 0;
const int OUT_SIDE = 1;
const int NOT_ENERGY = 2;
const int NOT_GOLD = 3;

const int D_NONE = -1;
const int D_LEFT = 0;
const int D_RIGHT = 1;
const int D_UP = 2;
const int D_DOWN = 3;
const int A_FREE = 4;
const int A_CRAFT = 5;

const std::string CMD[6] = {"0", "1", "2", "3", "4", "5"};

const int DX[4] = {0, 0, -1, 1};
const int DY[4] = {-1, 1, 0, 0};

const int MAX_W = 28+1;
const int MAX_H = 12+1;

#endif
