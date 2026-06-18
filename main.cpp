#include <iostream>
#include "test_utils.h"

// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main()
{
  std::cout<<"test arange ..."<<std::endl;
  test_arange();
  std::cout<<"done"<<std::endl;

  std::cout<<"test meshgrid ..."<<std::endl;
  test_meshgrid();
  std::cout<<"done"<<std::endl;

  std::cout<<"test sample grid ..."<<std::endl;
  test_samplegrid();
  std::cout<<"done"<<std::endl;

}