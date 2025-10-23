#ifndef PLAYER_H
#define PLAYER_H

class player
{
private:
public:
    int id, x, y, E, score, state, rest_count;
    player(int id, int x, int y, int E);
    void show();
};

#endif
