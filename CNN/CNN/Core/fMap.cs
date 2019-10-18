using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNN.Core;
using NN.Core;

namespace CNN.Core
{
    public enum Direction
    {
        Down,
        Left,
        Right,
        Up
    }

    public enum Notifier
    {
        Reader,
        Writer
    }

    public class fMap : Matrix<Node<double>>
    {
        private Position rd;
        private Position wr;

        private IReader reader;
        private IWriter writer;
        private int size;
        protected int bias;

        public fMap(int rows, int cols)
        {
            this.size = rows;
            Configure(rows, cols);
            Linalg linalg = new Linalg();

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i][j] = new Node<double>(linalg.RandomGaussian(0, 0.047855339));

            rd = new Position(0, 0, 0, 0);
            wr = new Position(0, 0, 0, 0);
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

        public void LockResetPosition()
        {

        }

        public bool Move(Notifier notifer, Direction d)
        {
            bool next = true;

            switch (d)
            {
                case Direction.Down:
                    // compute next
                    // if next == true move return true
                    // else return false
                    break;
                case Direction.Left:
                    break;
                case Direction.Right:
                    break;
                case Direction.Up:
                    break;
            }

            return next;
        }

        public void Next()
        {

        }

        public double[] Read()
        {
            throw new NotImplementedException();
        }

        public IReader Reader
        {
            set
            {
                reader = value;
                value.Source = this;
            }
        }

        public void Reset()
        {

        }

        public void Write(double val)
        {

        }

        public IWriter Writer
        {
            set
            {
                writer = value;
                value.Target = this;
            }
        }

        public struct Position
        {
            private int row, col;
            private int rrow, rcol;

            public Position(int row, int col, int rrow, int rcol)
            {
                this.row = row; this.col = col; this.rrow = rrow; this.rcol = rcol;
            }

            public int Col
            {
                get { return col; }
                set { col = value; }
            }

            public int[] Current
            {
                get { return new int[] { row, col }; }
                set { row = value[0]; col = value[1]; }
            }

            public int Row
            {
                get { return row; }
                set { row = value; }
            }

            public int[] Default
            {
                get { return new int[] { rrow, rcol }; }
            }
        }

        public int Size
        {
            get { return size; }
        }

        public int Bias
        {
            get { return bias; }
            set { bias = value; }
        }

        public Node<double>[][] value
        {
            get { return m.ToArray(); }

        }

    }
}
