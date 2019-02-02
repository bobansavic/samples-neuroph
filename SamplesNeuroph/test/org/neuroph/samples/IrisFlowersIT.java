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

import java.util.List;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import static org.neuroph.samples.BanknoteIT.trainingSet;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Nevena Miilenkovic
 */
public class IrisFlowersIT {

    static DataSet trainingSet;
    static DataSet testSet;

    public IrisFlowersIT() {
    }

    @BeforeClass
    public static void setUpClass() {
        String trainingSetFileName = "irisdatanormalised.txt";
        int inputsCount = 4;
        int outputsCount = 3;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, ",");
        dataSet.shuffle();

        List<DataSet> subSets = dataSet.split(60, 40);
        trainingSet = subSets.get(0);
        testSet = subSets.get(1);
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    @Test
    public void testMaxIterations() {
        int inputsCount = 4;
        int outputsCount = 3;
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, inputsCount, 2, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max iterations
        learningRule.setLearningRate(0.2);
        learningRule.setMaxIterations(1000);

        // train the network with training set
        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getCurrentIteration() <= learningRule.getMaxIterations());
    }

    @Test
    public void testMaxError() {
        int inputsCount = 4;
        int outputsCount = 3;
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, inputsCount, 2, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max iterations
        learningRule.setLearningRate(0.2);
        learningRule.setMaxError(0.03);

        // train the network with training set
        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getTotalNetworkError() <= learningRule.getMaxError());
    }

}
