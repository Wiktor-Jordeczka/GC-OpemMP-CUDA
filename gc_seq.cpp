#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <cstring>
#include <chrono>
using namespace std;

typedef struct Specimen {
    int* colors; // pointer do przechowywania kolorów
    int numOfColors = 0;
    int numOfConflicts = 0;

    bool operator< (const Specimen& other) const { // pomoc do sortowania
        if (numOfConflicts != other.numOfConflicts)
            return this->numOfConflicts < other.numOfConflicts;
        return this->numOfColors < other.numOfColors;
    }
} Specimen;

/*bool myCompare(const Specimen& a, const Specimen& b) { // pomoc do sortowania
    if (a.numOfConflicts != b.numOfConflicts)
        return a.numOfConflicts < b.numOfConflicts;
    return a.numOfColors < b.numOfColors;
}

void sortPopulation(Specimen* population, int populationSize) {
    sort(population, population + populationSize, myCompare);
}*/


/*class Specimen { // klasa osobnika (pokolorowania grafu)
public:
    int* colors;
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
};*/

void calculateFitness(int numOfVertices, int** adjacencyMatrix, Specimen &specimen) // sprawdzanie jakości rozwiązania
{
    specimen.numOfConflicts = 0;
    bool* colorSet = new bool[numOfVertices]; // użyte kolory
    std::memset(colorSet, 0, numOfVertices * sizeof(bool));

    for (int i = 0; i < numOfVertices; i++)
    {
        for (int j = i + 1; j < numOfVertices; j++) // sprawdzamy tylko od wierzchołka do konca zakresu, by uniknąć powtórzeń
        {
            if (adjacencyMatrix[i][j] == 1 && specimen.colors[i] == specimen.colors[j]) // szukamy wierzchołka sąsiedniego z tym samym kolorem
            {
                specimen.numOfConflicts++;
            }
        }
        //cout << "TEST "<< i<< " a " <<specimen.colors[i] <<endl;
        colorSet[specimen.colors[i]] = true; // kolor użyty
    }

    while (specimen.numOfConflicts == 0) { //Poprawiamy tylko dla rozwiązań bezkonfliktowych
        int maxColor = numOfVertices - 1;
        while (maxColor >= 0 && !colorSet[maxColor]) {
            maxColor--;
        }

        bool allColorsPresent = true; // sprawdzamy czy brakuje jakiegoś koloru
        for (int i = 0; i <= maxColor; i++) {
            if (!colorSet[i]) { // poprawiamy kolory
                allColorsPresent = false;
                int missingColor = i;

                // Obniżamy kolory o 1
                for (int j = 0; j < numOfVertices; j++) {
                    if (specimen.colors[j] > missingColor) {
                        specimen.colors[j]--;
                    }
                }

                colorSet[missingColor] = true;
                colorSet[maxColor] = false;
                break; // Restart sprawdzania
            }
        }

        if (allColorsPresent) { // wszystko ok
            break;
        }
    }

    specimen.numOfColors = 0; // ustawiamy liczbę kolorów
    for (int i = 0; i < numOfVertices; i++) {
        if (colorSet[i]) {
            specimen.numOfColors++;
        }
    }

    delete[] colorSet; // zwalniamy pamięć
    return;
}

/*vector<Specimen> tournament_selection(vector<Specimen> chromosomes, int population) // wybór nowej populacji metodą turniejową (starą zastępujemy wylosowaną)
{
    vector<Specimen> new_chromosomes; // deklaracja nowej populacji
    for (int i = 0; i < 2; i++) // robimy turniej dwa razy, gdyż po jednej iteracji otrzymamy 'populacja / 2' osobników
    {
        auto rng = default_random_engine{};
        shuffle(chromosomes.begin(), chromosomes.end(), rng); // wymieszanie osobników w starej populaji
        for (int j = 0; j < population; j += 2)
        {
            if (chromosomes[j].numOfConflicts <= chromosomes[j + 1].numOfConflicts) // porównywujemy sąsiadów i wybieramy tego "lepszego"
                new_chromosomes.push_back(chromosomes[j]);
            else
                new_chromosomes.push_back(chromosomes[j + 1]);
        }
    }
    return new_chromosomes;
}*/

float random_float(float max) // generowanie losowej liczby typu float z zakresu [0,max]
{
    float rng = static_cast<float> (rand()) / (static_cast <float> (RAND_MAX / max));
    return rng;
}

/*Specimen RouletteWheel_Selection(vector<Specimen> population, int populationSize) // koło fortuny
{
    float fitness_sum = 0;
    for (int i = 0; i < populationSize; i++)
    {
        fitness_sum += 1 / (1 + (float)(chromosomes[i].numOfConflicts)); //każdemu osobnikowi przydzielamy jego szanse na wybranie, mniej złych krawędzi = wieksze szanse na wybranie
        fitness_numbers.push_back(fitness_sum); //wektor z przedziałami każdego osobnika
    }
    while (new_chromosomes.size() < populationSize)
    {
        float rng = random_float(fitness_sum); // losujemy liczbę z zakresu [0; fitness_sum]
        for (int j = 0; j < fitness_numbers.size(); j++)
            if (rng <= fitness_numbers[j])
            {
                new_chromosomes.push_back(chromosomes[j]); // wylosowanego osobnika dołączamy do nowej populacji
                break;
            }
    }
    return chosenSpecimen;
}*/



int randomNumber(int min, int max) { // zwraca losową liczbę naturalną z zakresu [min;max]
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uni(min, max - 1);
    return uni(rng);
}

int getRandomNumber(int min, int max) { // if cuda breaks
    return min + rand() % ((max + 1) - min);
}

Specimen tournamentSelection(Specimen* population, int populationSize) { // turniej
    int tournamentSize = 3;
    Specimen chosenSpecimen = population[randomNumber(0, populationSize - 1)];
    for (int i = 1; i < tournamentSize; ++i) {
        Specimen candidate = population[randomNumber(0, populationSize - 1)];
        /*cout <<" cand "<< candidate.numOfColors<<endl;
        cout <<" cand "<< candidate.numOfConflicts<<endl;
        cout <<" ch "<< chosenSpecimen.numOfColors<<endl;
        cout <<" ch "<< chosenSpecimen.numOfConflicts<<endl;*/
        if (candidate < chosenSpecimen) {
            chosenSpecimen = candidate;
        }
    }
    return chosenSpecimen;
}

/*void crossover(Specimen& s1, Specimen& s2, int numOfVertices) {  // Krzyżowanie
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
}*/

// Krzyżowanie
void crossover(Specimen& parent1, Specimen& parent2, Specimen& offspring1, Specimen& offspring2, int numOfVertices) {
    int pivot = randomNumber(0, numOfVertices - 1);
    for (int i = 0; i < numOfVertices; ++i) {
        if (i <= pivot) {
            offspring1.colors[i] = parent1.colors[i];
            offspring2.colors[i] = parent2.colors[i];
        }
        else {
            offspring1.colors[i] = parent2.colors[i];
            offspring2.colors[i] = parent1.colors[i];
        }
    }
}

/*vector<Specimen> crossover2(vector<Specimen> chromosomes, int numOfVertices) //Krzyżowanie z dołączaniem dzieci do starej populacji
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
            childs[i].colors.push_back(childs[i + 1].colors.at(j));
        }
        childs[i + 1].colors.resize(pivot);
        for (int j = 0; j < numOfVertices - pivot; j++) { // wrzucamy część osobnika 1 z pomocniczej do osobnika 2
            childs[i + 1].colors.push_back(temp[j]);
        }
        chromosomes.push_back(childs[i]);
        chromosomes.push_back(childs[i + 1]);
    }
    return chromosomes;
}*/

/*Specimen mutateNew(Specimen s, int numOfColors, int numOfVertices) {  // Mutacja, nowy osobnik
    for (int i = 0; i < numOfVertices; i++) {
        s.colors[i] = randomNumber(0, numOfColors);
    }
    return s;
}*/

void mutateOld(Specimen& s, int numOfColors, int numOfVertices) {  // Mutacja, modyfikacja starego osobnika
    s.colors[randomNumber(0, numOfVertices)] = randomNumber(0, numOfColors); // mutacja losowego wierzchołka
}

/*void mutateOld2(Specimen& s, int numOfColors) {  // Mutacja, modyfikacja starego osobnika
    for (set<int>::iterator i = s.conflict.begin(); i != s.conflict.end(); i++) // mutacja wierzchołków konfliktowych
    {
        s.colors[*i] = randomNumber(0, numOfColors);
    }
}*/

vector<Specimen> populationElimination(vector<Specimen> chromosomes, const int population) { // Eliminacja populacji
    // Wybieramy najlepszych do zachowania
    int numOfBest = population / 2; // ilość najlepszych
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

void correct(Specimen* population, int numOfColors, int numOfVertices, int populationSize)
{ //Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
    for (int i = 0; i < populationSize; i++)
        if (population[i].numOfColors != numOfColors)
            for (int j = 0; j < numOfVertices; j++)
                if (population[i].colors[j] >= numOfColors - 1)
                    population[i].colors[j] = randomNumber(0, numOfColors - 2);
}

void initializePopulation(Specimen* population, int populationSize, int numOfVertices) {
    for (int i = 0; i < populationSize; ++i) {
        population[i].colors = (int*)malloc(numOfVertices * sizeof(int));
        for (int j = 0; j < numOfVertices; ++j) {
            population[i].colors[j] = randomNumber(0, numOfVertices - 1);
            //cout << i<<" pop " <<population[i].colors[j] <<endl;
        }
    }
}

#define inFile "queen7_7.txt" // Plik wejściowy
#define outFile "result_gc_seq.txt"
//nazwy plików: le450_5a.txt  gc500.txt  gc1000.txt  miles250.txt  queen6.txt  myciel7.txt  le450_25a.txt queen13.txt
#define printInterval 100 // co ile generacji wykonać print

int main()
{
    const int populationSize = 100; // ustawienie całkowitej populacji
    const int random_vertices = 30; // ilość losowo pokolorowanych wierzchołków przy tworzeniu populacji
    const int iterations = 5000; // Maksymalna liczba iteracji (generacji)
    int mutationChance = 60; // Tutaj wpisujemy prawdopodobieństwo mutacji <0;100>
    int stopTime = 60*5; // Maksymalny czas działania
    
    random_device random_generator; // generator do losowania liczb
    int numOfVertices; // ilość wierzchołków
    ifstream sourceFile(inFile); // plik wejściowy
    sourceFile >> numOfVertices;
    //const int random_vertices = numOfVertices/10; // alt
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

    //initializePopulation(population)
    //Specimen* population = new Specimen[populationSize];

    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen));
    initializePopulation(population, populationSize, numOfVertices);

    /*for (int i = 0; i < populationSize; i++) // wybieramy populację początkową
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
        Specimen test = Specimen();
        cout << "TEST" <<test.numOfColors <<endl;
        cout << "TEST" <<test.numOfConflicts <<endl;
        Specimen* test2 = new Specimen();
        cout << "TEST" <<test2->numOfColors <<endl;
        cout << "TEST" <<test2->numOfConflicts <<endl;

        population[i] = Specimen();
            population[i].colors = new int[numOfVertices];
        for (int i = 0; i < numOfVertices; i++) {
            population[i].colors = &vertexColor[i];
            cout << i<<" VC " <<vertexColor[i] <<endl;
            cout << i<<" pop " <<*population[i].colors <<endl;
        }
        cout << "TEST" <<population[i].colors[0] <<endl;
        cout << "TEST" <<population[i].colors[1] <<endl;
        cout << "TEST" <<population[i].colors[2] <<endl;
        //chromosomes.push_back(Specimen(numOfVertices, vertexColor)); // wpisanie wyników do wektora w obiekcie
        for (int j = 0; j < numOfVertices; j++) // resetowanie tablicy z pokolorowanymi wierzchołkami
            vertexColor[j] == -1;
        specimen =
        calculateFitness(numOfVertices, adjacencyMatrix, population[i]); // sprawdzanie jakości rozwiązania
    }*/
    for (int i = 0; i < populationSize; i++) {
        calculateFitness(numOfVertices, adjacencyMatrix, population[i]); // sprawdzanie jakości rozwiązania
    }
    /*for (int i = 0; i < populationSize; i++) {
        cout << i<<" a "<<population[i].numOfColors<<" a "<<population[i].numOfConflicts<<endl;
    }*/
    //sortPopulation(population, populationSize);
    sort(population, population+populationSize);
    /*for (int i = 0; i < populationSize; i++) {
        cout << i<<" a "<<population[i].numOfColors<<" a "<<population[i].numOfConflicts<<endl;
    }*/
    Specimen solution = Specimen(population[0]); // zmienna przetrzymująca optymalne rozwiązanie

    // PĘTLA 
    int iteration = 0;
    int numOfColors = 0;
    auto start = chrono::steady_clock::now(); // Timer
    auto stop = chrono::steady_clock::now(); // zatrzymanie po czasie
    while (iteration < iterations && chrono::duration_cast<chrono::seconds>(stop-start).count() < stopTime) {
        /*cout << "pętla" << endl;
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }*/
        //Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
        numOfColors = solution.numOfColors - 1;
        //cout << "numOfColors " << numOfColors<<endl;
        /*population = */
        correct(population, numOfColors, numOfVertices, populationSize);

        /*cout << "po korekcie" << endl;
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }*/

        if (iteration % printInterval == 0)
            cout << endl << "Generacja: " << iteration << " Aktualnie poszukuje rozwiazania dla: " << numOfColors << " kolorow, aktualna liczba konfliktow: " << population[0].numOfConflicts;

        //chromosomes = RouletteWheel_Selection(chromosomes, population); 
        //chromosomes = tournament_selection(chromosomes); // alt
        Specimen* newPopulation = (Specimen*)malloc(populationSize * sizeof(Specimen)); // Nowa populacja
        initializePopulation(newPopulation, populationSize, numOfVertices);

        // Turnieje i krzyżowanie (wybór nowej populacji)
        for (int i = 0; i < populationSize; i+=2) {
            Specimen parent1 = tournamentSelection(population, populationSize);
            Specimen parent2 = tournamentSelection(population, populationSize);
            Specimen offspring1;
            offspring1.colors = (int*)malloc(numOfVertices * sizeof(int));
            Specimen offspring2;
            offspring2.colors = (int*)malloc(numOfVertices * sizeof(int));
            crossover(parent1, parent2, offspring1, offspring2, numOfVertices);
            newPopulation[i] = offspring1;
            newPopulation[i + 1] = offspring2;
            /*if(i==98)
             cout << "yes! "<<iteration<<endl;*/
        }
        //chromosomes = crossover2(chromosomes, numOfVertices); //cross przez dodanie dzieci do populacji
        /*int curPopSize = chromosomes.size(); // Zapisujemy obecny rozmiar wektora do zmiennej aby się nie zapętliło
        for (int i = 0; i < chromosomes.size()-1; i = i+2) {
            crossover(chromosomes[i], chromosomes[i + 1], numOfVertices);
        }*/

        // MUTACJE
        //int curPopSize = populationSize; // Zapisujemy obecny rozmiar wektora do zmiennej aby się nie zapętliło
        /*for (int i = 0; i < populationSize; i++) {
            if (randomNumber(0, 100) < mutationChance) {
                mutateOld2(chromosomes[i], numOfColors); //mutacja przez zmienienie osobnika 
            }
        }*/
        for (int i = 0; i < populationSize; i++) {
            if (randomNumber(0, 100) < mutationChance) {
                mutateOld(newPopulation[i], numOfColors, numOfVertices); //mutacja przez zmienienie osobnika 
            }
        }

        /*cout << endl << endl;
        for (int i = 0; i < populationSize; i++) {
            cout << "Osobnik " << i << " konfliktow: " << newPopulation[i].numOfConflicts << " kolorow: " << newPopulation[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << newPopulation[i].colors[j] << " ";
            }
            cout << endl;
        }*/

        for (int i = 0; i < populationSize; i++) {
            calculateFitness(numOfVertices, adjacencyMatrix, newPopulation[i]); // sprawdzanie jakości rozwiązania
        }

        /*cout << "przed kopiowaniem" << endl;
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << newPopulation[i].numOfConflicts << " kolorow: " << newPopulation[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << newPopulation[i].colors[j] << " ";
            }
            cout << endl;
        }*/
        for (int i = 0; i < populationSize; ++i) {
            memcpy(population[i].colors, newPopulation[i].colors, numOfVertices * sizeof(int));
            population[i].numOfColors = newPopulation[i].numOfColors;
            population[i].numOfConflicts = newPopulation[i].numOfConflicts;
        }
        /*cout << "po kopiowaniu" << endl;
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << newPopulation[i].numOfConflicts << " kolorow: " << newPopulation[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << newPopulation[i].colors[j] << " ";
            }
            cout << endl;
        }*/


        for (int i = 0; i < populationSize; ++i) { // zwalnianie pamięci
            free(newPopulation[i].colors);
        }
        free(newPopulation);

        // NIEPOTRZEBNE W TEJ WERSJI
        // ELIMINACJA POPULACJI
        /*if (chromosomes[0].numOfConflicts > 4)
            chromosomes = populationElimination(chromosomes, populationSize);
        for (int i = 0; i < chromosomes.size(); i++) {
            chromosomes[i] = calculateFitness(numOfVertices, adjacencyMatrix, chromosomes[i]); // sprawdzanie jakości rozwiązania
        }*/
        
        sort(population, population+populationSize); // sortujemy tablicę tak, że najlepsze rozwiązanie ma indeks 0, a najgorsze 'populacja - 1'
        if (solution.numOfColors > population[0].numOfColors && population[0].numOfConflicts == 0)
            solution = Specimen(population[0]); // zapamiętujemy rozwiązanie, gdy znajdziemy rozwiązanie z mniejszą ilością kolorów
        iteration++;
        stop = chrono::steady_clock::now();

        /*cout << endl << endl;
        for (int i = 0; i < populationSize; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }*/
        /*cout << "po sortowaniu" << endl;
        cout << endl << endl;
        for (int i = 0; i < 2; i++) {
            cout << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                cout << population[i].colors[j] << " ";
            }
            cout << endl;
        }
        cout<<iteration<<endl;*/
    }

    // Wypisanie osobników na końcu
    fstream output(outFile, ios::out);
    output << "osobnik w zmiennej solution: " << " konfliktow: " << solution.numOfConflicts << " kolorow: " << solution.numOfColors << endl;
    for (int j = 0; j < numOfVertices; j++) {
        output << solution.colors[j] << " ";
    }
    output << endl << endl;
    for (int i = 0; i < populationSize; i++) {
        output << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
        for (int j = 0; j < numOfVertices; j++) {
            output << population[i].colors[j] << " ";
        }
        output << endl;
    }
    cout << endl << "Czas w sekuncach: " << chrono::duration_cast<chrono::seconds>(stop-start).count() << endl;
    delete[] population;
}