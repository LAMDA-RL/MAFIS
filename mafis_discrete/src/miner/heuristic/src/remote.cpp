#include "remote.h"
#include <sstream>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <iostream>

#define BUFFER_SIZE 80000

remote::remote(std::string HOST, int16_t PORT, int64_t timeout) {
    // Create socket
    this->socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (this->socket_fd < 0)
    {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        exit(1);
    }
 
    // Fill in server IP address
    struct sockaddr_in server;
    bzero( &server, sizeof( server ) );
  
    // Resolve server address (convert from symbolic name to IP number)
    struct hostent *host = gethostbyname(HOST.c_str());
    if (host == NULL)
    {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        exit(1);
    }
 
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
	
    // Write resolved IP address of a server to the address structure
    memmove(&(server.sin_addr.s_addr), host->h_addr_list[0], 4);
 
    // Connect to the remote server
    int res = connect(this->socket_fd, (struct sockaddr*) &server, sizeof(server));
    if (res < 0)
    {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        exit(1);
    }
 
    std::cout << "Connected. Reading a server message" << std::endl;
}

void remote::send_all(std::string message) {
    int n_bytes = write(this->socket_fd, message.c_str(), message.length());
    if(n_bytes < 0) {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        exit(1);
    }
    #ifdef DEBUG
    std::cout << "Sent: " << message.length() << std::endl;
    std::cout << "Sent message: " << message << std::endl;
    #endif
}

std::string remote::read_all() {
    char buffer[BUFFER_SIZE];
    bzero(buffer, BUFFER_SIZE);
    int n_bytes = read(this->socket_fd, buffer, BUFFER_SIZE);
    if (n_bytes < 0) {
        std::cerr << "Error: " << strerror(errno) << std::endl;
        exit(1);
    }
    #ifdef DEBUG
    std::cout << "Read: " << strlen(buffer) << std::endl;
    #endif
    return std::string(buffer);
}