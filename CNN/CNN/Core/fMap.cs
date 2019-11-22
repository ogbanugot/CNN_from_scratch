using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNN.Core;
using NN.Core;

namespace CNN.Core
{
   
    public class fMap : Matrix<Node<double>>
    {
      
        private int size;
        protected double[][] gradient;
        protected double[][] induced_field;
        protected double []bias = new double[2];

        public fMap(int rows, int cols)
        {
            this.size = rows;
            Configure(rows, cols);
            Linalg linalg = new Linalg();

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    m[i][j] = new Node<double>(linalg.RandomGaussian(0, 0.0047855339));
                }

            //gradient
            gradient = new double[rows][];
            for (int i = 0; i < rows; i++)
                gradient[i] = new double[cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    gradient[i][j] = 0;

            //Induced field
            induced_field = new double[rows][];
            for (int i = 0; i < rows; i++)
                induced_field[i] = new double[cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    induced_field[i][j] = 0;
        }

        public fMap Initialize(string path)
        {
            // todo
            //switch (filetype)
            //{

            //}
            return Initialize();
        }

        public fMap Initialize()
        {
            return this;
        }
               
        public int Size
        {
            get { return size; }
        }

        public double []Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        public Node<double>[][] value
        {
            get { return m.ToArray(); }

        }

        public double [][]Gradient
        {
            get { return gradient; }

            set
            {
                gradient = value;
            }
        }

        public double[][] InducedField
        {
            get { return induced_field; }

            set
            {
                induced_field = value;
            }
        }

    }
}
