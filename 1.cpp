#include <iostream>
int main(){
  *(char*)0 = 3;
  std::cout<<"A"<<std::endl;
  return 0;
}