using System;
using System.Linq;
using NN.Core;
using System.Collections.Generic;


namespace CNN.Core
{
    public abstract class Matrix<T>
        where T : new()
    {
        protected T[][] m;

        public Matrix()
        {
            ;
        }

        public Matrix(int rows, int cols)
        {
            Configure(rows, cols);
        }

        public Matrix<T> Configure(int rows, int cols)
        {
            m = new T[rows][];

            for (int i = 0; i < rows; i++)
                m[i] = new T[cols];

            return this;
        }

        public T Point(int i, int j)
        {
            return m[i][j];
        }

    }

    public class Linalg
    {
        public Filter MatrixProduct(Filter matrixA, Filter matrixB)
        {
            int aRows = matrixA.value.Length; int aCols = matrixA.value[0].Length;
            int bRows = matrixB.value.Length; int bCols = matrixB.value[0].Length;
            if (aCols != bRows)
                throw new Exception("Non-conformable matrices in MatrixProduct");

            Filter result = new Filter(null, aRows, 1);

            for (int i = 0; i < aRows; ++i) // each row of A
                for (int j = 0; j < bCols; ++j) // each col of B
                    for (int k = 0; k < aCols; ++k) // could use k < bRows
                        result.value[i][j].Value += matrixA.value[i][k].Value * matrixB.value[k][j].Value;

            return result;
        }

        public double Sum(Filter matrix)
        {

            double total = 0;
            for (int i = 0; i < matrix.value.Length; i++)
            {
                for (int j = 0; j < matrix.value[0].Length; j++)
                {
                    total += matrix.value[i][j].Value;
                }
            }
            return total;
        }

        public Node<double> Max(Filter matrix)
        {
            int Rows = matrix.value.Length; int Cols = matrix.value[0].Length;
            Node<double> max = new Node<double>(0);
            for (int i = 0; i < Rows; ++i)
            {
                for(int j = 0; j < Cols; ++j)
                {
                    if(matrix.value[i][j].Value > max.Value)
                    {
                        max = matrix.value[i][j];
                    }
                }
            }
            return max;

        }

        public double[][] MatrixCreate(int rows, int cols)
        {
            // allocates/creates a matrix initialized to all 0.0. assume rows and cols > 0
            // do error checking here
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];

            //for (int i = 0; i < rows; ++i)
            //  for (int j = 0; j < cols; ++j)
            //    result[i][j] = 0.0; // explicit initialization needed in some languages

            return result;
        }

        public IList<fMap> Flatten(fMap fmap)
        {
            var rows = fmap.value.Length;
            var cols = fmap.value[0].Length;
            IList<fMap> feature_maps = new List<fMap>();
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    fMap feature_map = new fMap(1, 1);
                    feature_map.value[0][0].Value = fmap.value[i][j].Value;
                    feature_maps.Add(feature_map);
                }
            return feature_maps;
        }

        public double RandomGaussian(double mean, double stdDev)
        {
            Random rand = new Random(); //reuse this if you are generating many
            double u1 = 1.0 - rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + stdDev * randStdNormal;
            return randNormal;
        }
    }
}