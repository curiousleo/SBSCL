/*
 * $Id$
 * $URL$
 * ---------------------------------------------------------------------
 * This file is part of Simulation Core Library, a Java-based library
 * for efficient numerical simulation of biological models.
 *
 * Copyright (C) 2007-2016 jointly by the following organizations:
 * 1. University of Tuebingen, Germany
 * 2. Keio University, Japan
 * 3. Harvard University, USA
 * 4. The University of Edinburgh, UK
 * 5. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 * 6. The University of California, San Diego, La Jolla, CA, USA
 * 7. The Babraham Institute, Cambridge, UK
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation. A copy of the license
 * agreement is provided in the file named "LICENSE.txt" included with
 * this software distribution and also available online as
 * <http://www.gnu.org/licenses/lgpl-3.0-standalone.html>.
 * ---------------------------------------------------------------------
 */
package org.simulator.sbml.astnode;

import org.sbml.jsbml.ASTNode;

/**
 * This class can compute and store the value of an integer node.
 * 
 * @author Roland Keller
 * @version $Rev$
 */
public class IntegerValue extends ASTNodeValue {

  /**
   * @param interpreter
   * @param node
   */
  public IntegerValue(ASTNodeInterpreter interpreter, ASTNode node) {
    super(interpreter, node);
  }

  /* (non-Javadoc)
   * @see org.simulator.sbml.astnode.ASTNodeValue#computeDoubleValue()
   */
  @Override
  protected void computeDoubleValue(double delay) {
    doubleValue = interpreter.compile(real, units);
  }

  /* (non-Javadoc)
   * @see org.simulator.sbml.astnode.ASTNodeValue#getConstant()
   */
  @Override
  public boolean getConstant() {
    return true;
  }

}
