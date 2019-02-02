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
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Ionosphere Dataset
    The Ionosphere Dataset requires the prediction of structure in the atmosphere given radar returns targeting free electrons in the ionosphere.
    The original data set that will be used in this experiment can be found at link: 
    https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data

2. Reference:  Space Physics Group , Applied Physics Laboratory ,Johns Hopkins University ,Johns Hopkins Road ,Laurel, MD 20723
   Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989). Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266. 

3. Number of instances: 351

4. Number of Attributes: 34 pluss class attributes

5. Attribute Information:    
   Inputs:
   34 attributes: 
   34 continuous features are computed for each radar return.
   Output: Class variable (0 or 1). Value 1 indicates good radar return.

6. Missing Values: None.



 
 */
public class Ionosphere implements LearningEventListener {

    public static void main(String[] args) {
        (new Ionosphere()).run();
    }

    public void run() {
        System.out.println("Creating training set...");
        // get path to training set
        String trainingSetFileName = "ionospheredata.txt";
        int inputsCount = 34;
        int outputsCount = 1;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, ",", false);
        Normalizer norm = new MaxNormalizer();
        norm.normalize(dataSet);
        dataSet.shuffle();

        List<DataSet> subSets = dataSet.split(60, 40);
        DataSet trainingSet = subSets.get(0);
        DataSet testSet = subSets.get(1);

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 30, 25, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener(this);

        // set learning rate and max error
        learningRule.setLearningRate(0.1);
        learningRule.setMaxError(0.01);
        System.out.println("Training network...");
        // train the network with training set
        neuralNet.learn(trainingSet);
        System.out.println("Training completed.");
        System.out.println("Testing network...");

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
            System.out.println(" Output: " + Arrays.toString(networkOutput));
            System.out.println("Desired output" + Arrays.toString(testSetRow.getDesiredOutput()));
        }
    }

    public void evaluate(NeuralNetwork neuralNet, DataSet dataSet) {
        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluateDataSet(neuralNet, dataSet);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.Binary.class);
        ConfusionMatrix confusionMatrix = evaluator.getResult();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics cm : metrics) {
            System.out.println(cm.toString() + "\r\n");
        }
        System.out.println(average.toString());
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
        MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }
}
