using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    class Loss
    {
        public double CategoricalCrossEntropy(List<double> probs, List<double> label)
        {
            if (probs.Count != label.Count)
                throw new Exception("Size mismatch: Label and probs should be the same ");
            double sum = 0;

            for (int i = 0; i < probs.Count; i++)
            {
                sum += label[i] * Math.Log(probs[i]);
            }
            return -sum;
        }
    }
}
