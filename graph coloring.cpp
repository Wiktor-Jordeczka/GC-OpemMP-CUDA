#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <chrono>
using namespace std;

class Specimen {
public:
    vector<int> colors;
    int numOfColors = 0;
    int numOfConflicts = 0;
    float fitness = 0;
    Specimen(int numOfVertices, int arr[]) {
        for (int i = 0; i < numOfVertices; i++) {
            colors.push_back(arr[i]);
        }
    };
    bool operator< (const Specimen& other) const { // pomoc do sortowania
        if (numOfConflicts != other.numOfConflicts)
            return this->numOfConflicts < other.numOfConflicts;
        return this->numOfColors < other.numOfColors;
    }
};


Specimen check_fitness(int numOfVertices, int** adjacencyMatrix, Specimen chromosomes) // sprawdzanie jakości rozwiązania
{
    chromosomes.numOfConflicts = 0;
    set<int> colorSet;
    for (int i = 0; i < numOfVertices; i++)
    {
        for (int j = i + 1; j < numOfVertices; j++) // sprawdzamy tylko od wierzchołka do konca zakresu, by uniknąć powtórzeń
        {
            if (adjacencyMatrix[i][j] == 1 && chromosomes.colors[i] == chromosomes.colors[j]) // szukamy wierzchołka sąsiedniego z tym samym kolorem
            {
                chromosomes.numOfConflicts++;
            }
        }
        colorSet.insert(chromosomes.colors[i]);
    }
    while (chromosomes.numOfConflicts == 0 && colorSet.size() != *(colorSet.rbegin()) + 1) //Poprawiamy tylko dla rozwiązań bezkonfliktowych
    {
        int missingColor;
      //  cout << "Wykryto niepoprawne uzycie kolorow\n";
        for (int i = 0; i < *(colorSet.rbegin()) + 1; i++)
            if (!(colorSet.count(i)))
            {
                missingColor = i; // znalezienie koloru, którego brakuje
                break;
            }
        for (int i = 0; i < numOfVertices; i++)
            if (chromosomes.colors[i] > missingColor) // Od każdego koloru większego od brakującego odejmujemy 1
                chromosomes.colors[i]--;

        colorSet.insert(missingColor);
        colorSet.erase(*(colorSet.rbegin())); // Poprawiamy Set by móc na nowo sprawdzić czy brakuje jakiegos koloru
    }
    chromosomes.numOfColors = colorSet.size();
    return chromosomes;
}

vector<Specimen> tournament_selection(vector<Specimen> chromosomes) // wybór nowej populacji metodą turniejową (starą zastępujemy wylosowaną)
{
    vector<Specimen> new_chromosomes; // deklaracja nowej populacji
    for (int i = 0; i < 2; i++) // robimy turniej dwa razy, gdyż po jednej iteracji otrzymamy 'populacja / 2' osobników
    {
        auto rng = default_random_engine{};
        shuffle(chromosomes.begin(), chromosomes.end(), rng); // wymieszanie osobników w starej populaji
        for (int j = 0; j < chromosomes.size(); j += 2)
        {
            if (chromosomes[j].numOfConflicts <= chromosomes[j + 1].numOfConflicts) // porównywujemy sąsiadów i wybieramy tego "lepszego"
                new_chromosomes.push_back(chromosomes[j]);
            else
                new_chromosomes.push_back(chromosomes[j + 1]);
        }
    }
    return new_chromosomes;
}

vector<Specimen> tournament_selection2(vector<Specimen> chromosomes) // wybór nowej populacji metodą turniejową (do nowej populacji dołączamy wylosowaną);
{
    vector<Specimen> new_chromosomes; // deklaracja nowej części populacji
    for (int i = 0; i < 2; i++) // robimy turniej dwa razy, gdyż po jednej iteracji otrzymamy 'populacja / 2' osobników
    {
        auto rng = default_random_engine{};
        shuffle(chromosomes.begin(), chromosomes.end(), rng); // wymieszanie osobników w starej populaji
        for (int j = 0; j < chromosomes.size(); j += 2)
        {
            if (chromosomes[j].numOfConflicts <= chromosomes[j + 1].numOfConflicts) // porównywujemy sąsiadów i wybieramy tego "lepszego"
                new_chromosomes.push_back(chromosomes[j]);
            else
                new_chromosomes.push_back(chromosomes[j + 1]);
        }
    }
    for (int i = 0; i < new_chromosomes.size(); i++)
        chromosomes.push_back(new_chromosomes[i]); //dołączanie wylosowanej populacji do starej populacji
    return chromosomes;
}


float random_float(float max) // generowanie losowej liczby typu float z zakresu [0,max]
{
    float rng = static_cast<float> (rand()) / (static_cast <float> (RAND_MAX / max));
    //cout << "Limit- " << max << " ";
    //cout << "Wylosowano: " << rng << endl;
    return rng;
    //random_device rd;
    //mt19937 rng(rd());
    //uniform_real_distribution<float> uni(0, max);
    //return uni(rng);
}

vector<Specimen> RouletteWheel_Selection(vector<Specimen> chromosomes)
{
    vector<Specimen> new_chromosomes;
    float fitness_sum = 0;
    vector<float> fitness_numbers;
    for (int i = 0; i < chromosomes.size(); i++)
    {
        fitness_sum += 1 / (1 + (float)(chromosomes[i].numOfConflicts)); //każdemu osobnikowi przydzielamy jego szanse na wybranie, mniej złych krawędzi = wieksze szanse na wybranie
        fitness_numbers.push_back(fitness_sum); //wektor z przedziałami każdego osobnika
    }
    while (new_chromosomes.size() < 50)
    {
        float rng = random_float(fitness_sum); // losujemy liczbę z zakresu [0; fitness_sum]
        for (int j = 0; j < fitness_numbers.size(); j++)
            if (rng <= fitness_numbers[j])
            {
                new_chromosomes.push_back(chromosomes[j]); // wylosowanego osobnika dołączamy do nowej populacji
               // cout << "Wylosowano osobnika " << j << "rng = " << rng << endl;
                break;
            }
    }
    return new_chromosomes;
}

int randomNumber(int min, int max) {
    // szybsze, ale mniej losowe

    //return min + (rand() % static_cast<int>(max - min + 1));

    // wolniejsze, ale bardziej losowe

    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uni(min, max - 1);
    return uni(rng);
}

void crossover(Specimen& s1, Specimen& s2, int numOfVertices){  // Krzyżowanie
    int pivot = randomNumber(1, numOfVertices - 1);  // losujemy pivot
    int* temp = new int[numOfVertices - pivot];  // tablica pomocnicza
    for (int i = numOfVertices - 1; i >= pivot; i--) {
        temp[i - pivot] = s1.colors.back(); // wrzucamy część osobnika 1 do pomocniczej
        s1.colors.pop_back();
    }
    for (int i = pivot; i < numOfVertices; i++) {  // wrzucamy część osobnika 2 do osobnika 1
        s1.colors.push_back(s2.colors.at(i));
    }
    s2.colors.resize(pivot);
    for (int i = 0; i < numOfVertices - pivot; i++) { // wrzucamy część osobnika 1 z pomocniczej do osobnika 2
        s2.colors.push_back(temp[i]);
    }
}

vector<Specimen> crossover2(vector<Specimen> chromosomes, int numOfVertices) //Krzyżowanie z dołaczaniem dzieci do starej populacji
{
    vector<Specimen> childs = chromosomes;
    for (int i = 0; i < childs.size(); i += 2)
    {
        int pivot = randomNumber(1, numOfVertices - 1);  // losujemy pivot
        int* temp = new int[numOfVertices - pivot];  // tablica pomocnicza
        for (int j = numOfVertices - 1; j >= pivot; j--) {
            temp[j - pivot] = childs[i].colors.back(); // wrzucamy część osobnika 1 do pomocniczej
            childs[i].colors.pop_back();
        }
        for (int j = pivot; j < numOfVertices; j++) {  // wrzucamy część osobnika 2 do osobnika 1
            childs[i].colors.push_back(childs[i+1].colors.at(j));
        }
        childs[i+1].colors.resize(pivot);
        for (int j = 0; j < numOfVertices - pivot; j++) { // wrzucamy część osobnika 1 z pomocniczej do osobnika 2
            childs[i+1].colors.push_back(temp[j]);
        }
        chromosomes.push_back(childs[i]);
        chromosomes.push_back(childs[i + 1]);
    }
    return chromosomes;
}

Specimen mutateNew(Specimen s, int numOfColors, int numOfVertices) {  // Mutacja, nowy osobnik
    s.colors[randomNumber(0, numOfVertices)] = randomNumber(0, numOfColors);
    return s;
}

/*Specimen mutateNew(Specimen s, int numOfColors, int numOfVertices) {  // Mutacja, nowy osobnik
    for(int i=0;i<numOfVertices;i++){
        s.colors[i] = randomNumber(0, numOfColors);
    }
    return s;
}*/

void mutateOld(Specimen& s, int numOfColors, int numOfVertices) {  // Mutacja, modyfikacja starego osobnika
    s.colors[randomNumber(0, numOfVertices)] = randomNumber(0, numOfColors);
    /*for(int i=0;i<numOfVertices;i++){
        if(randomNumber(0,100)<20){
            s.colors[i] = randomNumber(0, numOfColors);
        }
    }*/
}

vector<Specimen> populationElimination(vector<Specimen> chromosomes, const int population) { // Eliminacja populacji
    // Wybieramy najlepszych do zachowania
    int numOfBest = population/2; // ilość najlepszych; nienawidzę liczb nieparzystych
    vector<Specimen> bestSpecimen = chromosomes;
    sort(bestSpecimen.begin(), bestSpecimen.end());
    vector<Specimen> otherSpecimen; // pozostali na później
    for (int i = bestSpecimen.size(); i > numOfBest; i--) {
        otherSpecimen.push_back(bestSpecimen.back());
        bestSpecimen.pop_back();
    }

    // Wybieramy losowo z pozostałych
    vector<Specimen> randomSpecimen;
    randomSpecimen.reserve(population - numOfBest);
    for (int i = 0; i < population - numOfBest; i++) {
        int num = randomNumber(0, otherSpecimen.size() - 1);
        randomSpecimen.push_back(otherSpecimen[num]);
        otherSpecimen.erase(otherSpecimen.begin() + num);
    }
    vector<Specimen> result;
    result.reserve(bestSpecimen.size() + randomSpecimen.size());
    result.insert(result.end(), bestSpecimen.begin(), bestSpecimen.end());
    result.insert(result.end(), randomSpecimen.begin(), randomSpecimen.end());
    return result;
}

vector<Specimen> correct(vector<Specimen> chromosomes, int numOfColors, int numOfVertices)
{
    for (int i = 0; i < chromosomes.size(); i++)
        if (chromosomes[i].numOfColors != numOfColors)
            for (int j = 0; j < numOfVertices; j++)
                if (chromosomes[i].colors[j] >= numOfColors - 1)
                    chromosomes[i].colors[j] = randomNumber(0, numOfColors - 2);
    return chromosomes;
}

#define inFile "queen6.txt" // Plik wejściowy

int main()
{
    //Parametry
    const int population = 50; // ustawienie całkowitej populacji
    const int random_vertices = 5; // ilość losowo pokolorowanych wierzchołków przy tworzeniu populacji
    const int iterations = 10000000; // Ile generacji
    int mutationChance = 10; // Tutaj wpisujemy startowe prawdopodobieństwo mutacji <0;100>
    int nextMutationChance = 40; // Tutaj wpisujemy późniejsze prawdopodobieństwo mutacji <0;100>
    int mutationChange = 5000; //Tutaj wpisujemy numer generacji, po której zwiększamy szansę mutacji
    int stopTime = 30; //Tutaj czas stopu w sekundach

    random_device random_generator; // generator do losowania liczb
    int numOfVertices; // ilość wierzchołków
    ifstream sourceFile(inFile); // plik wejściowy
    sourceFile >> numOfVertices;
    int** adjacencyMatrix = new int* [numOfVertices];
    for (int i = 0; i < numOfVertices; i++)
        adjacencyMatrix[i] = new int[numOfVertices] {}; // macierz adjacencji
    int a, b; // para wierzchołków
    while (sourceFile >> a >> b) { // wczutujemy dane z pliku
        adjacencyMatrix[a - 1][b - 1] = 1;
        adjacencyMatrix[b - 1][a - 1] = 1;
    }
    sourceFile.close();
    int* vertexColor = new int [numOfVertices] {}; // kolory wierzchołków
    bool* colorsUsed = new bool [numOfVertices] {}; // pomocnicza

    vector<Specimen> chromosomes; // definicja zbioru populacji 

    for (int i = 0; i < population; i++) // wybieramy populację początkową
    {
        fill(vertexColor, vertexColor + numOfVertices, -1);
        for (int j = 0; j < random_vertices; j++)
        {
            uniform_int_distribution<int> random_number(0, numOfVertices); // losujemy jeden, dowolny wierzchołek
            vertexColor[random_number(random_generator)] = j; // Wylosowanym wierzchołkom przypisujemy kolejno różne kolory
        }

        for (int i = 0; i < numOfVertices; i++) { // sprawdzamy jakie kolory nie mogą być użyte
            if (vertexColor[i] == -1) { // kolorujemy tylko jeszcze niepokolorowane wierzchołki
                for (int j = 0; j < numOfVertices; j++) {
                    if (adjacencyMatrix[i][j]) {
                        if (vertexColor[j] != -1) {
                            colorsUsed[vertexColor[j]] = true;
                        }
                    }
                }
                for (int j = 0; j < numOfVertices; j++) { // wybieramy kolor
                    if (!(colorsUsed[j])) {
                        vertexColor[i] = j;
                        break;
                    }
                }
                for (int j = 0; j < numOfVertices; j++) { // reset tablicy pomocniczej
                    if (adjacencyMatrix[i][j]) {
                        if (vertexColor[j] != -1) {
                            colorsUsed[vertexColor[j]] = false;
                        }
                    }
                }
            }
        }
        chromosomes.push_back(Specimen(numOfVertices, vertexColor)); // wpisanie wyników do wektora w obiekcie
        for (int j = 0; j < numOfVertices; j++) // resetowanie tablicy z pokolorowanymi wierzchołkami
            vertexColor[j] == -1;
        chromosomes[i] = check_fitness(numOfVertices, adjacencyMatrix, chromosomes[i]); // sprawdzanie jakości rozwiązania
    }

    sort(chromosomes.begin(), chromosomes.end());
    Specimen solution = chromosomes[0]; // zmienna przetrzymująca optymalne rozwiązanie

    // Wypisanie osobników na początku
    /*for (int i = 0; i < chromosomes.size(); i++) {
        cout << "Osobnik " << i << " konfliktow: " << chromosomes[i].numOfConflicts << " kolorow: " << chromosomes[i].numOfColors << endl;
        for (int j = 0; j < chromosomes[i].colors.size(); j++) {
            cout << chromosomes[i].colors[j] << " ";
        }
        cout << endl;
    }*/

    // PĘTLA 
    auto start = chrono::steady_clock::now(); // Timer
    auto stop = chrono::steady_clock::now(); // zatrzymywanie po czasie;
    int iteration = 0;
    int numOfColors = 0; // A tu nie lepiej num of Vertices?
    while (iteration < iterations && chrono::duration_cast<chrono::seconds>(stop-start).count() < stopTime) {
        if(iteration==mutationChange){
            mutationChance = nextMutationChance;
        }

        numOfColors = solution.numOfColors;
        for (int i = 0; i < population; i++) // Znalezienie aktualnego, najlepszego pokolorowania
            if (chromosomes[i].numOfColors < numOfColors && chromosomes[i].numOfConflicts == 0)
                numOfColors = chromosomes[i].numOfColors;
        numOfColors--; // Odejmujemy 1, gdyż kolorujemy od 0

        chromosomes = correct(chromosomes, numOfColors, numOfVertices);
        //auto stop = chrono::steady_clock::now(); // zatrzymywanie po czasie;

        if (iteration % 100 == 0)
            cout << endl << "Generacja: " << iteration << " Aktualna liczba kolorow: " << numOfColors + 1<< " sekund: " << chrono::duration_cast<chrono::seconds>(stop-start).count();

        

        //chromosomes = tournament_selection2(chromosomes); // Wybieranie nowej populacji
        //chromosomes = tournament_selection(chromosomes); // alt
        chromosomes = RouletteWheel_Selection(chromosomes);

        // KRZYŻOWANIE

        //for (int i = 0; i < chromosomes.size(); i += 2) {
        //    crossover(chromosomes[i], chromosomes[i + 1], numOfVertices);
        //}
        chromosomes = crossover2(chromosomes, numOfVertices); //cross przez dodanie dzieci o populacji
      
        // MUTACJE
        //int mutationChance = iteration/(iterations/100);
        int curPopSize = chromosomes.size(); // Zapisujemy obecny rozmiar wektora do zmiennej aby się nie zapętiło
        for (int i = 0; i < curPopSize; i++) {
            if (randomNumber(0, 100) < mutationChance) {
               //chromosomes.push_back(mutateNew(chromosomes[i], numOfColors, numOfVertices)); //mutacja poprzez dodanie wyniku do populacji
               mutateOld(chromosomes[i], numOfColors, numOfVertices); //mutacja przez zmienienie osobnika 
            }
        }
       

        // ELIMINACJA POPULACJI
        chromosomes = populationElimination(chromosomes, population);
        //cout<<endl<<chromosomes.size();
        for (int i = 0; i < chromosomes.size(); i++) {
            chromosomes[i] = check_fitness(numOfVertices, adjacencyMatrix, chromosomes[i]); // sprawdzanie jakości rozwiązania
        }
        sort(chromosomes.begin(), chromosomes.end()); // sortujemy tablicę tak, że najlepsze rozwiązanie ma indeks 0, a najgorsze 'populacja - 1'

        if (solution.numOfColors > chromosomes[0].numOfColors && chromosomes[0].numOfConflicts == 0)
            solution = chromosomes[0]; // zapamiętujemy rozwiązanie, gdy znajdziemy rozwiązanie z mniejszą ilością kolorów
        iteration++;

        stop = chrono::steady_clock::now(); // zatrzymywanie po czasie;
    }

    // Wypisanie osobników na końcu
    cout << endl;
    for (int i = 0; i < chromosomes.size(); i++) {
        cout << "Osobnik " << i << " konfliktow: " << chromosomes[i].numOfConflicts << " kolorow: " << chromosomes[i].numOfColors << endl;
        for (int j = 0; j < chromosomes[i].colors.size(); j++) {
            cout << chromosomes[i].colors[j] << " ";
        }
        cout << endl;
    }
    cout << endl << "osobnik w zmiennej solution: " << " konfliktow: " << solution.numOfConflicts << " kolorow: " << solution.numOfColors << endl;
    for (int j = 0; j < solution.colors.size(); j++) {
        cout << solution.colors[j] << " ";
    }
    cout<<endl;
}