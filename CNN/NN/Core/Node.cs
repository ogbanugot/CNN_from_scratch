using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public class Node<T>
    {
        private static int globalID = 0;

        public Node() { }

        public Node(T t)
        {
            Value = t;
        }

        public Node(int? id, T t)
        {
            ID = id;
            Value = t;
        }

        public int? ID { get; set; }

        public static int NextGlobalD
        {
            get { return globalID++; }
        }

        public static void ResetGlobalID()
        {
            globalID = 0;
        }

        public T Value { get; set; }
    }
}