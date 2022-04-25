#include <omp.h>
#include <iostream>
using namespace std;
int main() {
    char hello_string[] = "Hello World from thread";
    // Значение «1234» смысловой нагрузки не имеет и используется
    // для того, чтобы показать, что id у всех потоков
    // в параллельной области отличается от id до параллельной
    // области
    int id = 1234;
    int size = 0;
    cout << "Initial thread: id = " << id << ", size = " << size << endl;
    // Директива OpenMP: объявление параллельной области
    #pragma omp parallel private(id) shared(hello_string)
    {
        int size = omp_get_num_threads();
        cout << "Number of threads = " << size << endl;
        cout.flush();
        id = omp_get_thread_num();
        cout << hello_string << " " << id << endl;
        cout.flush();
    } // Конец параллельной области
    return 0;
}