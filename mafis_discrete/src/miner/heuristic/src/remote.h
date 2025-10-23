#ifndef REMOTE_H
#define REMOTE_H

#include <string>

class remote {
    private:
        int socket_fd;

    public:
        remote(std::string HOST, int16_t PORT, int64_t timeout);
        void send_all(std::string data);
        std::string read_all();
};

#endif