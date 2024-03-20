import * as tf from '@tensorflow/tfjs';

// Step 1 - build Convolutional network
export const buildCnn = function (data: any) {
  return new Promise<{ model: tf.Sequential, data: any }>((resolve, reject) => {
    const model = tf.sequential();

    model.add(tf.layers.conv1d({
      inputShape: [data.dates.length, 1],
      kernelSize: 100,
      filters: 8,
      strides: 2,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling1d({
      poolSize: [500],
      strides: [2]
    }));

    model.add(tf.layers.conv1d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling1d({
      poolSize: [100],
      strides: [2]
    }));

    model.add(tf.layers.dense({
      units: 10,
      kernelInitializer: 'VarianceScaling',
      activation: 'softmax'
    }));

    resolve({ model, data });
  });
}

// Step 2 Train Model
export const cnn = function (model: tf.Sequential, data: any, cycles: number) {
  const tdates = tf.tensor1d(data.dates);
  const thighs = tf.tensor1d(data.highs);
  const test = tf.tensor1d(data.test_times);

  return new Promise<void>((resolve, reject) => {
    setTimeout(() => {
      try {
        const optimizer = tf.train.sgd(0.1); // Learning rate set here
        model.compile({ optimizer, loss: 'binaryCrossentropy' });
        model.fit(
          tdates.reshape([1, 1960, 1]),
          thighs.reshape([1, 1960, 1]),
          {
            batchSize: 3,
            epochs: cycles
          }
        ).then(() => {
          console.log('');
          console.log(`Running CNN for AAPL at ${cycles} epochs`);
          const prediction = model.predict(test) as tf.Tensor;
          if (Array.isArray(prediction)) {
            console.error('Prediction result is an array');
            resolve();
          } else {
            console.log(prediction.dataSync());
            console.log(data.test_highs);
            resolve();
          }
        });
      } catch (ex) {
        console.error(ex);
        resolve();
      }
    }, 5000);
  });
}

