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
import org.neuroph.nnet.Adaline;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Nevena Milenkovic
 */
public class SwedishAutoInsuranceT {

    static DataSet trainingSet;
    static DataSet testSet;

    public SwedishAutoInsuranceT() {
    }

    @BeforeClass
    public static void setUpClass() {
        String dataSetFileName = "autodata.txt";
        int inputsCount = 1;
        int outputsCount = 1;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFileName, inputsCount, outputsCount, ",", false);
        Normalizer norm = new MaxNormalizer();
        norm.normalize(dataSet);
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
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();

        learningRule.setMaxIterations(1000);

        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getCurrentIteration() <= learningRule.getMaxIterations());

    }

    @Test
    public void testMaxError() {
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();

        learningRule.setMaxError(0.01);

        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getTotalNetworkError() < learningRule.getMaxError());
    }

}
