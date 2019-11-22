using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    class Actfunc
    {
        public double ReLu(double x)
        {
            if (x >= 0)
                return x;
            else
                return 0;
        }

        public double DReLuConn(double x)
        {
            if (x >= 0)
                return 1;
            else
                return 0;
        }

        public fMap DReLuConv(fMap dconv)
        {
            throw new NotImplementedException();
        }

        public List<double> Softmax(List<double> output)
        {
            List<double> vs = new List<double>();
            List<double> softmax = new List<double>();

            for (int i = 0; i < output.Count; i++)
            {
                vs.Add(Math.Exp(output[i]));
            }
            var total = vs.Sum();
            for (int i = 0; i < vs.Count; i++)
            {
                softmax.Add(vs[i] / total);
            }
            return softmax;
        }
    }
}
