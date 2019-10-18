using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using NN.Core;

namespace NN.Nodes
{
    public abstract class Neuron<T> : Node<T>
    {
        protected IList<Synapse<T>> synapse = new List<Synapse<T>>();

        public Neuron() { }

        public Neuron(T t)
            : base(t) { }

        public Neuron(int? id, T t)
            : base(id, t) { }

        public abstract void Drive();

        public Node<T> Source
        {
            set {
                Random rnd = new Random();
                synapse.Add(new Synapse<T>(value, rnd.NextDouble())); 
            }
        }

        public IList<Synapse<T>> Synapse
        {
            get { return synapse; }
        }
    }

    public struct Synapse<T>
    {
        private Node<T> source;
        private double weight;

        public Synapse(Node<T> source, double weight)
        {
            this.source = source;
            this.weight = weight;
        }

        public Node<T> Source
        {
            get { return source; }
        }
    }
}
