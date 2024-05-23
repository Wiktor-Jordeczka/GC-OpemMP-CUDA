#include <iostream>
#include <random>
#include <string>
#include <fstream>
#include <algorithm>

void create_edges(int* edges_list, int max_edges, int edges_num)
{
    for (int i = 0; i < max_edges; i++)
        if (edges_num > i)
        {
            edges_list[i] = 1;
        }
        else
            edges_list[i] = 0;
}

void connected_graph(int** matrix, int v)
{
    for (int i = 0; i < v; i++)
    {
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<int> dist6(0, v - 1);
        int num = dist6(rng);
        while (num == i || (matrix[num][i] == 1 || matrix[i][num] == 1))
        {
            num = dist6(rng);
            dist6.reset();
        }
        matrix[i][num] = 1;
        matrix[num][i] = 1;
        dist6.reset();
    }
}

void create_matrix(int** matrix, int* edges_list, int v)
{
    int mark = v - 1;
    for (int i = 0; i < v; i++)
        for (int j = 0; j < v; j++)
            if (matrix[i][j] != 1)
            {
                if (j > i)
                {
                    matrix[i][j] = edges_list[mark++];
                }
                else
                    matrix[i][j] = 0;
            }
}

void create_list(int** matrix, int**list, int v)
{
    int position = 0;
    for (int i = 0; i < v; i++)
        for (int j = 0; j < v; j++)
            if (j > i && matrix[i][j] == 1)
            {
                list[0][position] = i + 1;
                list[1][position] = j + 1;
                position++;
            }
}

void display_matrix(int** matrix, int v)
{
    for (int j = 0; j < v; j++) {
        for (int i = 0; i < v; i++) {
            std::cout << matrix[j][i] << ' ';
        }
        std::cout << std::endl;
    }
}

void display_list(int** list, int max_edges, int edges_num)
{
    for (int i = 0; i < edges_num; i++)
    {
        std::cout << list[0][i] << ", " << list[1][i] << std::endl;
    }
}

void saving(int** list, int v, int edges_num)
{
    std::fstream output("data1.txt", std::ios::out);

    output << v << std::endl;
    for (int i = 0; i < edges_num-1; i++)
    {
        output << list[0][i] << " " << list[1][i] << std::endl;
    }
    output << list[0][edges_num-1] << " " << list[1][edges_num-1];
    output.close();
}


int main()
{
    int v, max_edges, control, edges_num; // v-number of vertices, max_edges - number of maximum edges graph can have with v vertices, control - help variable, edges_num - number of expected edges in created graph
    float d; // density percentage
    int** matrix, **list; // matrix - used to generete random graph, list - contains graph as edge list
    int* edges_list; // used to randomize edges in a graph, help list
    do
    {
        std::cout << "Please type the number or vertices in graph: ";
        std::cin >> v;
    } while (v <= 1);
    max_edges = (v * v - v) / 2;
    do
    {
        std::cout << "Please type the % of density: ";
        std::cin >> d;
    } while (d >= 1 || round(d * max_edges) < v);
    control = v;
    edges_num = static_cast<int> (round(d * max_edges));
    edges_list = new int[max_edges];
    matrix = new int* [v];
    for (int i = 0; i < v; i++)
        matrix[i] = new int[v] { 0 };
    list = new int* [2];
    for (int i = 0; i < 2; i++)
        list[i] = new int[edges_num]{ 0 };
    //Creating graph edges
    create_edges(edges_list, max_edges, edges_num);

    //Shuffling edges
    unsigned seed = 0;
    std::shuffle(edges_list + v, edges_list + max_edges, std::default_random_engine(seed));

    //Making sure its a connected graph
    connected_graph(matrix, v);

    //Creating matrix
    create_matrix(matrix, edges_list, v);

    //Converting to edge list
    create_list(matrix, list, v);

    //Displaying the matrix
    //display_matrix(matrix, v);

    //Displaying the list
    //display_list(list, max_edges, edges_num);

    //Saving graph to a text file
    saving(list, v, edges_num);

}
