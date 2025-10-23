#include <iostream>
using namespace std;



struct step
{
    int x, y;
    int E;
    int prev_x, prev_y;

    step(int x, int y, int E) {
        this->x = x;
        this->y = y;
        this->E = E;
        this->prev_x = -1;
        this->prev_y = -1;
    }

    void set_prev(int x, int y) {
        this->prev_x = x;
        this->prev_y = y;
    }
    
    const bool operator == (const step &rhs) const {
        return E == rhs.E;
    }

    const bool operator < (const step &rhs) const {
        return E < rhs.E;
    }

    const bool operator >(const step &rhs) const {
        return E > rhs.E;
    }


};

int main() {
    step a(1, 1, 2);
    step b(1, 2, 3);
    if(a < b) cout << "OK";
}