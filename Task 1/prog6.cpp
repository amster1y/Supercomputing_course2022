#include <omp.h>
#include <stdio.h>
#include <iostream>
#define NT 4
using namespace std;
int main( )
{
    int section_count = 0;
    omp_set_num_threads(NT);
    #pragma omp parallel
    #pragma omp sections firstprivate(section_count)
    {
        #pragma omp section
        {
            section_count++;
            // Может быть напечатано 1 или 2
            cout << "section_count " << section_count << endl;
        }
        #pragma omp section
        {
            section_count++;
            // Может быть напечатано 1 или 2
            cout << "section_count " << section_count << endl;
        }
    }
    return 0;
}