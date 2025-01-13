import torch

class RTools:
    """
    
    @breif: A class for calculating R2 scores, dynamic divergence loss, and divergence metrics.
    @author: Niki Steffen Radinsky

    This part of my resarch it  calculate the R2 score, dynamic divergence loss, and divergence metrics.
    In older version of Pytorch this helped with performance by using the R2 score and dynamic divergence loss.
    The dynamic part helped with the speed in which the model learned and the R2 score helped with the accuracy of the model.
    Strangly in newer versions of Pytorch this is not needed and cause the opposite effect. Presumetly optimzation in the newer versions of Pytorch.
    Include something like this already indirectly through optimazation in the newer versions cause the opposite effect using this code.
    """
    # The R2 score is a statistical measure of how close the data are to the fitted regression line.
    """
    Takes in the input and target tensors and calculates the R2 score.
    @param input: The predicted values.
    @param target: The true values.
    @param multioutput: Strategy for calculating the R2 score.
    @param num_regressors: The number of regressors in the model.
    @return: The R2 score.
    """
    @staticmethod
    def r2_score(input, target, multioutput="uniform_average", num_regressors=0):
       
     
        # Calculate the mean of the target values
        mean_target = torch.mean(target, dim=0)

        # Calculate the total sum of squares
        total_sum_squares = torch.sum(torch.pow(target - mean_target, 2), dim=0)

        # Calculate the residual sum of squares
        residual_sum_squares = torch.sum(torch.pow(target - input, 2), dim=0)

        # Calculate the R2 score
        r2 = 1 - (residual_sum_squares / total_sum_squares)

        if multioutput == "uniform_average":
            # Average the R2 scores with uniform weight
            r2 = torch.mean(r2)
        elif multioutput == "variance_weighted":
            # Weight the R2 scores by the variances of each individual output
            output_variances = torch.var(target, dim=0, unbiased=True)
            r2 = torch.mean(r2 * output_variances)

        if num_regressors > 0:
            # Adjust the R2 score for the number of regressors
            n_samples = input.size(0)
            adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - num_regressors - 1))
            return adjusted_r2

        return r2
    # The dynamic divergence loss is a loss function that penalizes predictions that are too close to the true values.
    """
    Calculate the dynamic divergence loss for a set of predictions.
    @param input: The predicted values.
    @param target: The true values.
    @param dynamic: The dynamic parameter for the loss function.
    @param size_average: Whether to average the loss over the batch.
    @return: The dynamic divergence loss.
    """
    @staticmethod
    def r2_dd_loss(input, target, dynamic, size_average=True):
        
        
        r2 = RTools.r2_score(input, target)
        dd_l = torch.abs(r2 - 1) + 1
        s = torch.relu(1 / (r2 - 2))
        m = dynamic - dynamic * s
        dd_loss = torch.pow(dd_l, m) - 1

        if size_average:
            return torch.mean(dd_loss)
        else:
            return torch.sum(dd_loss)
    # The divergence metric is a measure of how much the predictions diverge from the true values.
    """
    Calculate the divergence metric between two tensors.
    @param x: The first tensor.
    @param y: The second tensor.
    """
    @staticmethod
    def divergence(x, y): 
       
        difference = torch.abs(x - y)
        percentage_difference = (difference / y) * 100
        num_items = percentage_difference.numel()
        return torch.sum(percentage_difference) / num_items
