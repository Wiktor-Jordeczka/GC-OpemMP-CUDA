#include <iostream>
#include <fstream>
#include <algorithm>
using namespace std;

#define inFile "gc500.txt"
#define outFile "result.txt"

int main()
{
    int numOfVertices; // ilość wierzchołków
    ifstream sourceFile(inFile); // plik wejściowy
    sourceFile >> numOfVertices;
    int adjacencyMatrix[numOfVertices][numOfVertices] = {}; // macierz adjacencji
    int a, b; // para wierzchołków
    while(sourceFile >> a >> b){ // wczutujemy dane z pliku
        adjacencyMatrix[a-1][b-1] = 1;
        adjacencyMatrix[b-1][a-1] = 1;
    }
    sourceFile.close();

    int vertexColor[numOfVertices]; // kolory wierzchołków
    fill(vertexColor, vertexColor+numOfVertices, -1);
    bool colorsUsed[numOfVertices] = {}; // pomocnicza

    for(int i=0; i<numOfVertices; i++){ // sprawdzamy jakie kolory nie mogą być użyte
        for(int j=0; j<numOfVertices; j++){
            if(adjacencyMatrix[i][j]){
                if(vertexColor[j] != -1){
                    colorsUsed[vertexColor[j]] = true;
                }
            }
        }
        for(int j=0; j<numOfVertices; j++){ // wybieramy kolor
            if(!(colorsUsed[j])){
                vertexColor[i] = j;
                break;
            }
        }
        for(int j=0; j<numOfVertices; j++){ // reset tablicy pomocniczej
            if(adjacencyMatrix[i][j]){
                if(vertexColor[j] != -1){
                    colorsUsed[vertexColor[j]] = false;
                }
            }
        }
    }

    ofstream resultFile(outFile);
    for(int i=0; i<numOfVertices; i++){ // wypisanie wyniku + zapis do pliku
        cout << "Wierzchołek " << i+1 << " kolor " << vertexColor[i]+1 << endl;
        resultFile << "Wierzchołek " << i+1 << " kolor " << vertexColor[i]+1 << endl;
    }

    int numOfColorsUsed = 0;
    for(int i=0; i<numOfVertices; i++){
        if(vertexColor[i] > numOfColorsUsed)
            numOfColorsUsed = vertexColor[i];
    }
    cout << endl << "Użyto " << numOfColorsUsed+1 << " kolorów" << endl;
    resultFile << "Użyto " << numOfColorsUsed+1 << " kolorów" << endl;
    resultFile.close();

}