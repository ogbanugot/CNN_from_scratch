using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core.Filters
{
    public class Maxpool : Filter
    {
        public Maxpool(int? id, int size, int stride)
            : base(id, size, stride) { }

        public int[] forward(int image, int f, int stride)
        {
            //forward propagation through maxpool layer
            //returns downsampled
            return null;
        }

        public int[] backward(int? dpool, int origin, int f,int stride)
        {
            //backprop through maxpool layer
            //returns dout
            return null;
        }
    }
}
