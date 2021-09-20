#include <stdio.h>
#include <stdlib.h>
#include "./incl/read.h"

int main(int argc, char const *argv[])
{
    char image_path[] = "simplehough1-256x256.raw";
    read_image(image_path);
    
    return 0;
}


