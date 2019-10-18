using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Nodes.Variants
{
    public class Perceptron : Neuron<double>
    {
        public Perceptron() { }

        public Perceptron(double d)
            : base(d) { }

        public Perceptron(int? id, double d)
            : base(id, d) { }

        public override void Drive()
        {
            throw new NotImplementedException();
        }
    }
}