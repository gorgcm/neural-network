use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use reqwest::blocking::Client;
use flate2::read::GzDecoder;

struct Random {
    state: u64,
}

impl Random {
    fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Random { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x1fffffffffffff) as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

#[inline]
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[inline]
    fn init_random(&mut self, rng: &mut Random, scale: f64) {
        for val in self.data.iter_mut() {
            *val = rng.next_gaussian() * scale;
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }

    fn add(&mut self, other: &Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);

        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    fn subtract(&mut self, other: &Matrix) {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);

        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
    }

    fn multiply(&mut self, scalar: f64) {
        for val in self.data.iter_mut() {
            *val *= scalar;
        }
    }

    fn dot(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(self.cols, other.rows);

        let mut result = Matrix::new(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                let row_offset = i * self.cols;
                for k in 0..self.cols {
                    sum += self.data[row_offset + k] * other.data[k * other.cols + j];
                }
                result.data[i * result.cols + j] = sum;
            }
        }

        result
    }

    fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j * result.cols + i] = self.data[i * self.cols + j];
            }
        }

        result
    }

    fn apply_function<F>(&mut self, func: F) where F: Fn(f64) -> f64 {
        for val in self.data.iter_mut() {
            *val = func(*val);
        }
    }

    fn apply_derivative(&self, derivative: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for (i, &val) in self.data.iter().enumerate() {
            result.data[i] = derivative(val);
        }
        result
    }

    fn hadamard_product(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(self.rows, other.rows);
        debug_assert_eq!(self.cols, other.cols);

        let mut result = Matrix::new(self.rows, self.cols);
        for (i, (&a, &b)) in self.data.iter().zip(other.data.iter()).enumerate() {
            result.data[i] = a * b;
        }
        result
    }

    fn clone(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(),
        }
    }
}

struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights1: Matrix,
    biases1: Matrix,
    weights2: Matrix,
    biases2: Matrix,
    z1: Matrix,
    a1: Matrix,
    z2: Matrix,
    a2: Matrix,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, rng: &mut Random) -> Self {
        let mut weights1 = Matrix::new(hidden_size, input_size);
        let mut biases1 = Matrix::new(hidden_size, 1);
        let mut weights2 = Matrix::new(output_size, hidden_size);
        let mut biases2 = Matrix::new(output_size, 1);

        let w1_scale = (2.0 / (input_size + hidden_size) as f64).sqrt();
        let w2_scale = (2.0 / (hidden_size + output_size) as f64).sqrt();

        weights1.init_random(rng, w1_scale);
        biases1.init_random(rng, 0.1);
        weights2.init_random(rng, w2_scale);
        biases2.init_random(rng, 0.1);

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights1,
            biases1,
            weights2,
            biases2,
            z1: Matrix::new(0, 0),
            a1: Matrix::new(0, 0),
            z2: Matrix::new(0, 0),
            a2: Matrix::new(0, 0),
        }
    }

    fn forward(&mut self, input: &Matrix) -> &Matrix {

        self.z1 = self.weights1.dot(input);

        for i in 0..self.z1.rows {
            let bias = self.biases1.data[i];
            for j in 0..self.z1.cols {
                self.z1.data[i * self.z1.cols + j] += bias;
            }
        }

        self.a1 = self.z1.clone();
        self.a1.apply_function(relu);

        self.z2 = self.weights2.dot(&self.a1);

        for i in 0..self.z2.rows {
            let bias = self.biases2.data[i];
            for j in 0..self.z2.cols {
                self.z2.data[i * self.z2.cols + j] += bias;
            }
        }

        self.a2 = self.z2.clone();
        self.a2.apply_function(sigmoid);

        &self.a2
    }

    fn backward(&mut self, input: &Matrix, target: &Matrix, learning_rate: f64) {
        let batch_size = input.cols as f64;

        let mut error_output = self.a2.clone();
        error_output.subtract(target);

        let a1_t = self.a1.transpose();
        let dw2 = error_output.dot(&a1_t);

        let mut db2 = Matrix::new(self.output_size, 1);
        for i in 0..self.output_size {
            let row_start = i * error_output.cols;
            let mut sum = 0.0;
            for j in 0..error_output.cols {
                sum += error_output.data[row_start + j];
            }
            db2.data[i] = sum;
        }

        let w2_t = self.weights2.transpose();
        let error_hidden = w2_t.dot(&error_output);

        let relu_derivative_z1 = self.z1.apply_derivative(relu_derivative);
        let error_hidden = error_hidden.hadamard_product(&relu_derivative_z1);

        let x_t = input.transpose();
        let dw1 = error_hidden.dot(&x_t);

        let mut db1 = Matrix::new(self.hidden_size, 1);
        for i in 0..self.hidden_size {
            let row_start = i * error_hidden.cols;
            let mut sum = 0.0;
            for j in 0..error_hidden.cols {
                sum += error_hidden.data[row_start + j];
            }
            db1.data[i] = sum;
        }

        let lr_batch = learning_rate / batch_size;

        let mut weights1_update = dw1;
        weights1_update.multiply(lr_batch);
        self.weights1.subtract(&weights1_update);

        let mut biases1_update = db1;
        biases1_update.multiply(lr_batch);
        self.biases1.subtract(&biases1_update);

        let mut weights2_update = dw2;
        weights2_update.multiply(lr_batch);
        self.weights2.subtract(&weights2_update);

        let mut biases2_update = db2;
        biases2_update.multiply(lr_batch);
        self.biases2.subtract(&biases2_update);
    }

    fn predict(&mut self, input: &Matrix) -> Vec<usize> {
        let output = self.forward(input);
        let mut predictions = Vec::with_capacity(output.cols);

        for j in 0..output.cols {
            let mut max_idx = 0;
            let mut max_val = output.get(0, j);

            for i in 1..output.rows {
                let val = output.get(i, j);
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            predictions.push(max_idx);
        }

        predictions
    }
}

struct MnistData {
    training_images: Matrix,
    training_labels: Matrix,
    test_images: Matrix,
    test_labels: Matrix,
}

fn read_u32(reader: &mut dyn Read) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn load_mnist_dataset(path: &Path) -> io::Result<MnistData> {
    println!("Loading MNIST dataset from {:?}", path);

    let mut file = File::open(path.join("train-images-idx3-ubyte"))?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2051, "Invalid magic number for training images");

    let num_images = read_u32(&mut file)? as usize;
    let rows = read_u32(&mut file)? as usize;
    let cols = read_u32(&mut file)? as usize;
    let pixels_per_image = rows * cols;

    println!("Training images: {} images of {}x{} pixels", num_images, rows, cols);

    let mut training_images = Matrix::new(pixels_per_image, num_images);
    let mut buffer = vec![0u8; pixels_per_image * num_images];
    file.read_exact(&mut buffer)?;

    for i in 0..num_images {
        for j in 0..pixels_per_image {
            training_images.set(j, i, buffer[i * pixels_per_image + j] as f64 / 255.0);
        }
    }

    let mut file = File::open(path.join("train-labels-idx1-ubyte"))?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2049, "Invalid magic number for training labels");

    let num_labels = read_u32(&mut file)? as usize;
    assert_eq!(num_labels, num_images, "Number of labels does not match number of images");

    let mut buffer = vec![0u8; num_labels];
    file.read_exact(&mut buffer)?;

    let mut training_labels = Matrix::new(10, num_labels);
    for i in 0..num_labels {
        let label = buffer[i] as usize;
        training_labels.set(label, i, 1.0);
    }

    let mut file = File::open(path.join("t10k-images-idx3-ubyte"))?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2051, "Invalid magic number for test images");

    let num_images = read_u32(&mut file)? as usize;
    let rows = read_u32(&mut file)? as usize;
    let cols = read_u32(&mut file)? as usize;
    let pixels_per_image = rows * cols;

    println!("Test images: {} images of {}x{} pixels", num_images, rows, cols);

    let mut test_images = Matrix::new(pixels_per_image, num_images);
    let mut buffer = vec![0u8; pixels_per_image * num_images];
    file.read_exact(&mut buffer)?;

    for i in 0..num_images {
        for j in 0..pixels_per_image {
            test_images.set(j, i, buffer[i * pixels_per_image + j] as f64 / 255.0);
        }
    }

    let mut file = File::open(path.join("t10k-labels-idx1-ubyte"))?;
    let magic = read_u32(&mut file)?;
    assert_eq!(magic, 2049, "Invalid magic number for test labels");

    let num_labels = read_u32(&mut file)? as usize;
    assert_eq!(num_labels, num_images, "Number of labels does not match number of images");

    let mut buffer = vec![0u8; num_labels];
    file.read_exact(&mut buffer)?;

    let mut test_labels = Matrix::new(10, num_labels);
    for i in 0..num_labels {
        let label = buffer[i] as usize;
        test_labels.set(label, i, 1.0);
    }

    Ok(MnistData {
        training_images,
        training_labels,
        test_images,
        test_labels,
    })
}

fn evaluate(nn: &mut NeuralNetwork, images: &Matrix, labels: &Matrix) -> f64 {
    let predictions = nn.predict(images);
    let mut correct = 0;

    for i in 0..predictions.len() {
        let mut actual_label = 0;
        for j in 0..10 {
            if labels.get(j, i) > 0.5 {
                actual_label = j;
                break;
            }
        }

        if predictions[i] == actual_label {
            correct += 1;
        }
    }

    correct as f64 / predictions.len() as f64
}

fn get_batch(
    images: &Matrix,
    labels: &Matrix,
    batch_size: usize,
    batch_idx: usize
) -> (Matrix, Matrix) {
    let start_idx = batch_idx * batch_size;
    let end_idx = (start_idx + batch_size).min(images.cols);
    let actual_size = end_idx - start_idx;

    let mut batch_images = Matrix::new(images.rows, actual_size);
    let mut batch_labels = Matrix::new(labels.rows, actual_size);

    for i in 0..actual_size {
        let src_idx = start_idx + i;
        for j in 0..images.rows {
            batch_images.set(j, i, images.get(j, src_idx));
        }
        for j in 0..labels.rows {
            batch_labels.set(j, i, labels.get(j, src_idx));
        }
    }

    (batch_images, batch_labels)
}

fn download_file(url: &str, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("Downloading {} to {:?}", url, path);

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let client = Client::new();
    let mut response = client.get(url).send()?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP status {}", response.status()).into());
    }

    let mut file = File::create(path)?;
    let mut buffer = [0u8; 8192];
    let mut bytes_written = 0;

    while let Ok(bytes_read) = response.read(&mut buffer) {
        if bytes_read == 0 { break; }
        file.write_all(&buffer[..bytes_read])?;
        bytes_written += bytes_read;
        print!("\rDownloaded: {} bytes", bytes_written);
        io::stdout().flush().ok();
    }
    println!();

    Ok(())
}

fn decompress_gzip(input_path: &Path, output_path: &Path) -> io::Result<()> {
    println!("Decompressing {:?} to {:?}", input_path, output_path);

    let input_file = File::open(input_path)?;
    let mut decoder = GzDecoder::new(input_file);
    let mut output_file = File::create(output_path)?;

    io::copy(&mut decoder, &mut output_file)?;
    println!("Decompression complete");

    Ok(())
}

fn ensure_mnist_data(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(path)?;

    let files = [
        ("train-images-idx3-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"),
        ("train-labels-idx1-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"),
        ("t10k-images-idx3-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"),
        ("t10k-labels-idx1-ubyte", "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"),
    ];

    for (file, url) in &files {
        let file_path = path.join(file);

        if !file_path.exists() {
            let gz_path = path.join(format!("{}.gz", file));
            download_file(url, &gz_path)?;
            decompress_gzip(&gz_path, &file_path)?;
            let _ = fs::remove_file(&gz_path); 
        } else {
            println!("File {:?} already exists, skipping download", file_path);
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Minimalist Neural Network MNIST Classifier");
    println!("=========================================");

    let mut rng = Random::new();
    let input_size = 28 * 28;  
    let hidden_size = 128;
    let output_size = 10;      
    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size, &mut rng);

    let mnist_path = PathBuf::from("./data");
    println!("Checking for MNIST dataset...");

    if !mnist_path.exists() || 
       !mnist_path.join("train-images-idx3-ubyte").exists() ||
       !mnist_path.join("train-labels-idx1-ubyte").exists() ||
       !mnist_path.join("t10k-images-idx3-ubyte").exists() ||
       !mnist_path.join("t10k-labels-idx1-ubyte").exists() {
        println!("MNIST dataset not found. Downloading now...");
        ensure_mnist_data(&mnist_path)?;
    }

    println!("Loading dataset...");
    let data = match load_mnist_dataset(&mnist_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load MNIST dataset: {}", e);
            ensure_mnist_data(&mnist_path)?;
            load_mnist_dataset(&mnist_path)?
        }
    };

    let epochs = 10;
    let batch_size = 64;
    let learning_rate = 0.01;
    let num_batches = (data.training_images.cols + batch_size - 1) / batch_size;

    println!("Training for {} epochs with batch size {}", epochs, batch_size);
    println!("Network: {}-{}-{}", input_size, hidden_size, output_size);

    let start_time = Instant::now();

    for epoch in 0..epochs {
        let epoch_start = Instant::now();

        for batch_idx in 0..num_batches {
            let (batch_images, batch_labels) = get_batch(
                &data.training_images,
                &data.training_labels,
                batch_size,
                batch_idx
            );

            nn.forward(&batch_images);
            nn.backward(&batch_images, &batch_labels, learning_rate);

            if batch_idx % 100 == 0 {
                print!("\rEpoch {}/{}, Batch {}/{}", epoch + 1, epochs, batch_idx, num_batches);
                io::stdout().flush().ok();
            }
        }

        let accuracy = evaluate(&mut nn, &data.test_images, &data.test_labels);
        println!("\rEpoch {}/{}: {:.2}% accuracy in {:?}", 
                epoch + 1, epochs, accuracy * 100.0, epoch_start.elapsed());
    }

    let final_accuracy = evaluate(&mut nn, &data.test_images, &data.test_labels);
    println!("\nTraining completed in {:?}", start_time.elapsed());
    println!("Final test accuracy: {:.2}%", final_accuracy * 100.0);

    Ok(())
}
