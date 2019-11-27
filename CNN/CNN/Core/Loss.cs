using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    class Loss
    {
        public double CategoricalCrossEntropy(double[][] probs, double[][] label)
        {
            if (probs[0].Count() != label[0].Count())
                throw new Exception("Size mismatch: Label and probs should be the same ");
            double sum = 0;

            for (int i = 0; i < probs[0].Count(); i++)
            {
                sum += label[0][i] * Math.Log(probs[0][i]);
            }
            return -sum;
        }
    }
}
