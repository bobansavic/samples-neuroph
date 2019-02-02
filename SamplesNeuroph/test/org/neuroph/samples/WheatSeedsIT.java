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
import static org.neuroph.samples.IonosphereIT.trainingSet;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Nevena Milenkovic
 */
public class WheatSeedsIT {

    static DataSet trainingSet;
    static DataSet testSet;

    public WheatSeedsIT() {
    }

    @BeforeClass
    public static void setUpClass() {
        String trainingSetFileName = "seeds.txt";
        int inputsCount = 7;
        int outputsCount = 3;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, "\t");
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
        int inputsCount = 7;
        int outputsCount = 3;
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 15, 10, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max iterations
        learningRule.setLearningRate(0.1);
        learningRule.setMaxIterations(1000);

        // train the network with training set
        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getCurrentIteration() <= learningRule.getMaxIterations());
    }

    @Test
    public void testMaxError() {
        int inputsCount = 7;
        int outputsCount = 3;
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 15, 10, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max iterations
        learningRule.setLearningRate(0.1);
        learningRule.setMaxError(0.1);

        // train the network with training set
        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getTotalNetworkError() <= learningRule.getMaxError());
    }

}
