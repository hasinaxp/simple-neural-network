/*
 * test : 
 *      a simple implementation of neural network on xor example
 * 
 */



#include <iostream>
#include "neural_network.hpp"
#include <vector>
#include <cstdio>


int main()
{
    // creating neural network
    // 2 input neurons, 3 hidden neurons and 1 output neuron 
    std::vector<uint32_t> topology = {2,3,1};
    sp::SimpleNeuralNetwork nn(topology, 0.1);
    
    //sample dataset
    std::vector<std::vector<float>> targetInputs = {
        {0.0f, 0.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    }; 
    std::vector<std::vector<float>> targetOutputs = {
        {0.0f},
        {0.0f},
        {1.0f},
        {1.0f}
    };

    uint32_t epoch = 100000;
    
    //training the neural network with randomized data
    std::cout << "training start\n";

    for(uint32_t i = 0; i < epoch; i++)
    {
        uint32_t index = rand() % 4;
        nn.feedForword(targetInputs[index]);
        nn.backPropagate(targetOutputs[index]);
    }

    std::cout << "training complete\n";


    //testing the neural network
    for( std::vector<float> input : targetInputs)
    {
        nn.feedForword(input);
        std::vector<float> preds = nn.getPredictions();
        std::cout << input[0] << "," << input[1] <<" => " << preds[0] << std::endl;
    }

    return 0;
}
