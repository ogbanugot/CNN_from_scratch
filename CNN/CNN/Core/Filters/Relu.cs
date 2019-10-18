using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core.Filters
{
    public class Relu : Filter
    {
        public Relu(int? id, int size, int stride)
            : base(id, size, stride) { }
    }
}
