const tf = require('@tensorflow/tfjs-node');

// Define a simple linear regression model
class LinearRegressionModel {
    constructor(inputSize, outputSize) {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({ inputShape: [inputSize], units: outputSize }));
        this.model.compile({ optimizer: tf.train.sgd(0.01), loss: 'meanSquaredError' });
    }

    train(xTrain, yTrain, epochs) {
        return this.model.fit(xTrain, yTrain, { epochs });
    }

    predict(x) {
        return this.model.predict(x);
    }
}

// Create the model
const inputSize = 1;
const outputSize = 1;
const learningRate = 0.01;
const numEpochs = 100;

const model = new LinearRegressionModel(inputSize, outputSize);

// Generate sample data (replace this with your dataset)
const xTrain = tf.tensor2d([[1.0], [2.0], [3.0], [4.0]]);
const yTrain = tf.tensor2d([[2.0], [4.0], [6.0], [8.0]]);

// Training the model
model.train(xTrain, yTrain, numEpochs)
    .then(history => {
        console.log('Training history:', history);
        
        // Make predictions
        const newData = tf.tensor2d([[5.0]]);
        const prediction = model.predict(newData);
        prediction.print();
    })
    .catch(error => console.error('Error during training:', error));

// Save the model (optional)
async function saveModel() {
    await model.model.save('file://./linear_regression_model');
    console.log('Model saved.');
}

// Uncomment the line below to save the model
// saveModel();
