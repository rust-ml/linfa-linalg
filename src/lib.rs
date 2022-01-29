use linfa::prelude::SingleTargetRegression;
use linfa::traits::{Fit, Predict};
use linfa_linear::LinearRegression;

fn test() {
    let dataset = linfa_datasets::diabetes();
    let model = LinearRegression::default().fit(&dataset).unwrap();
    let pred = model.predict(&dataset);
    let r2 = pred.r2(&dataset).unwrap();
    println!("r2 from prediction: {}", r2);
}
