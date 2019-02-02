/**
 * Copyright 2013 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.samples;

import java.util.Arrays;
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.core.leraning.error.MeanAbsoluteError;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.nnet.Adaline;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Swedish Auto Insurance Dataset
    The Swedish Auto Insurance Dataset involves predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.
    The original data set that will be used in this experiment can be found at link: 
    https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt

2. Reference: Swedish Committee on Analysis of Risk Premium in Motor Insurance
 
3. Number of instances: 63

4. Number of Attributes: 2 (input is numerical, output is continuous)

5. Attribute Information:    
   In the following data
   X = number of claims (numerical)
   Y = total payment for all the claims in thousands of Swedish Kronor (continuous) for geographical zones in Sweden.


6. Missing Values: none



 
 */
public class SwedishAutoInsurance implements LearningEventListener {

    public static void main(String[] args) {
        (new SwedishAutoInsurance()).run();
    }

    public void run() {
        System.out.println("Creating training set...");
        String dataSetFileName = "autodata.txt";
        int inputsCount = 1;
        int outputsCount = 1;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFileName, inputsCount, outputsCount, ",", false);
        Normalizer norm = new MaxNormalizer();
        norm.normalize(dataSet);
        dataSet.shuffle();

        List<DataSet> subSets = dataSet.split(60, 40);
        DataSet trainingSet = subSets.get(0);
        DataSet testSet = subSets.get(1);

        System.out.println("Creating neural network...");
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();
        learningRule.addListener(this);

        // train the network with training set
        System.out.println("Training network...");
        neuralNet.learn(trainingSet);
        System.out.println("Training completed.");

        System.out.println("Network outputs for test set");

//testNeuralNetwork(neuralNet, testSet);
        evaluate(neuralNet, testSet);

        System.out.println("Saving network");
        // save neural network to file
        neuralNet.save("nn1.nnet");

        System.out.println("Done.");
    }

    public void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for (DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString(testSetRow.getInput()));
            System.out.println(" Output: " + networkOutput[0]);
            System.out.println("Desired output" + Arrays.toString(networkOutput));

        }
    }

    public void evaluate(NeuralNetwork neuralNet, DataSet dataSet) {
        MeanSquaredError mse = new MeanSquaredError();
        MeanAbsoluteError mae = new MeanAbsoluteError();

        for (DataSetRow testSetRow : dataSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();
            double[] desiredOutput = testSetRow.getDesiredOutput();
            mse.addPatternError(networkOutput, desiredOutput);
            mae.addPatternError(networkOutput, desiredOutput);
        }

        System.out.println("Mean squared error is: " + mse.getTotalError());
        System.out.println("Mean absolute error is: " + mae.getTotalError());
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        LMS bp = (LMS) event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }

}
