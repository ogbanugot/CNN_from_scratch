using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Core
{
    public interface IWriter
    {
        fMap Target { set; }

        int Stride { get; }
    }
}
