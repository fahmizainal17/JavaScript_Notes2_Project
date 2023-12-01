const tf = require('@tensorflow/tfjs-node');

/**
 * Linear Regression Model for TensorFlow.js in JavaScript.
 */
class LinearRegressionModel {
    /**
     * Constructor for the Linear Regression Model.
     * @param {number} inputSize - The size of the input features.
     * @param {number} outputSize - The size of the output predictions.
     */
    constructor(inputSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ inputShape: [inputSize], units: outputSize }));
        this.model.compile({ optimizer: tf.train.sgd(0.01), loss: 'meanSquaredError' });
    }

    /**
     * Train the linear regression model.
     * @param {tf.Tensor} xTrain - The input training data.
     * @param {tf.Tensor} yTrain - The output training data.
     * @param {number} epochs - The number of training epochs.
     * @returns {Promise} A promise that resolves with the training history.
     */
    train(xTrain, yTrain, epochs) {
        return this.model.fit(xTrain, yTrain, { epochs });
    }

    /**
     * Make predictions using the trained model.
     * @param {tf.Tensor} x - The input data for predictions.
     * @returns {tf.Tensor} The predictions.
     */
    predict(x) {
        return this.model.predict(x);
    }
}

/**
 * Main function to demonstrate linear regression training and prediction.
 */
async function main() {
    // Define model parameters
    const inputSize = 1;
    const outputSize = 1;
    const learningRate = 0.01;
    const numEpochs = 100;

    // Create the model
    const model = new LinearRegressionModel(inputSize, outputSize);

    // Generate sample data (replace this with your dataset)
    const xTrain = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
    const yTrain = tf.tensor2d([[2.0], [4.0], [6.0], [8.0]]);

    try {
        // Train the model
        const history = await model.train(xTrain, yTrain, numEpochs);
        console.log('Training history:', history);

        // Make predictions
        const newData = tf.tensor2d([[5.0]]);
        const prediction = model.predict(newData);
        prediction.print();
    } catch (error) {
        console.error('Error during training:', error);
    }

    // Save the model (optional)
    await saveModel();
}

/**
 * Save the trained model to disk (optional).
 * @returns {Promise} A promise that resolves when the model is saved.
 */
async function saveModel() {
    await model.model.save('file://./linear_regression_model');
    console.log('Model saved.');
}

// Run the main function
main();
