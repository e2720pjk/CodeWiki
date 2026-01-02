# Joern Integration User Guide

## Overview

CodeWiki now supports **Hybrid Analysis** that combines traditional AST parsing with Joern CPG (Code Property Graph) analysis for enhanced insights into data flow and cross-module dependencies.

## Quick Start

### Basic Usage (AST Only - Default)
```bash
codewiki generate
```

### Enhanced Analysis with Joern
```bash
codewiki generate --use-joern
```

### Full Feature Example
```bash
codewiki generate --use-joern --create-branch --github-pages --verbose
```

## What Joern Adds

### 1. Data Flow Analysis
- **Parameter-to-local variable flows**: Track how parameters are used within functions
- **Cross-function data dependencies**: Identify data sharing between functions
- **Variable usage patterns**: Understand local variable relationships

### 2. Enhanced Call Graphs
- **Cross-module relationship detection**: Better identification of dependencies between modules
- **Resolved vs unresolved calls**: Distinguish between actual and potential calls
- **Confidence scoring**: Quality metrics for relationship detection

### 3. Structural Insights
- **Control Flow Graphs (CFG)**: Understanding execution paths
- **Program Dependency Graphs (PDG)**: Data and control dependencies
- **Complexity metrics**: Function-level complexity analysis

## Requirements

### System Dependencies
- **Java 19+**: Required for Joern backend
- **Graphviz**: For visualization (optional but recommended)

### Installation
```bash
# Install Graphviz (macOS)
brew install graphviz

# Joern JAR is downloaded automatically when --use-joern is used
# Or manually place joern.jar in project root
```

## Performance

### Baseline Comparison
Based on our testing with typical repositories:

| Analysis Type | Speed | Insights | Reliability |
|-------------|-------|----------|-------------|
| AST Only | 1x (baseline) | Basic structure | ★★★★★ |
| Hybrid (AST + Joern) | 2-3x slower | Deep data flow | ★★★★☆ |

### When to Use Joern

#### ✅ Recommended For:
- **Medium repositories** (50-500 files)
- **Data-intensive applications** (need data flow insights)
- **Security analysis** (tracking data flow)
- **Complex microservices** (cross-module dependencies)

#### ❌ Not Recommended For:
- **Very large repositories** (1000+ files) - may be slow
- **Simple documentation** needs only
- **Production pipelines** requiring maximum speed

## Features

### 1. Feature Flags
- `--use-joern`: Enable Joern CPG analysis
- `--verbose`: Show detailed analysis progress

### 2. Graceful Fallback
If Joern fails (missing Java, incompatible files, etc.), the system automatically falls back to AST-only analysis:

```
⚠️  Joern analysis failed: Error: Unable to access jarfile joern.jar
🔄 Falling back to AST analysis
✓ Documentation generated successfully
```

### 3. Output Enhancement

#### Standard AST Output
```json
{
  "functions": [...],
  "relationships": [...],
  "summary": {
    "total_functions": 45,
    "total_relationships": 67
  }
}
```

#### Enhanced Joern Output
```json
{
  "functions": [...],
  "relationships": [...],
  "data_flow_relationships": [
    {
      "source": "process_data:user_input",
      "target": "process_data:processed_input",
      "flow_type": "parameter_to_local",
      "confidence": 0.8
    }
  ],
  "cross_module_relationships": [...],
  "summary": {
    "total_functions": 45,
    "total_relationships": 67,
    "data_flow_relationships": 12,
    "cross_module_relationships": 8
  },
  "metadata": {
    "analysis_time": 2.34,
    "joern_enhanced": true,
    "analysis_type": "hybrid_ast_joern"
  }
}
```

## Troubleshooting

### Common Issues

#### 1. "Joern JAR not found"
**Solution**: Let CodeWiki download it automatically or manually download joern.jar:

```bash
curl -L https://github.com/joernio/joern/releases/latest/download/joern.jar -o joern.jar
```

#### 2. "Java not found"
**Solution**: Install Java 19+:

```bash
# macOS
brew install openjdk@19

# Ubuntu/Debian
sudo apt install openjdk-19-jdk

# Verify installation
java --version
```

#### 3. "Analysis timeout"
**Solution**: For large repositories, consider:
- Using AST-only mode for initial analysis
- Increasing timeout limits (advanced)
- Splitting repository into smaller chunks

#### 4. "Memory errors"
**Solution**: Increase Java heap size:

```bash
export JAVA_OPTS="-Xmx4g"
codewiki generate --use-joern
```

### Debug Mode

Use verbose mode for detailed debugging:

```bash
codewiki generate --use-joern --verbose
```

This shows:
- Joern initialization status
- Analysis progress per file
- Performance metrics
- Error details

## Advanced Usage

### 1. Selective Analysis

For very large repositories, analyze specific directories:

```bash
# Focus on core modules
codewiki generate --use-joern --include="src/core/*,src/api/*"

# Exclude test files
codewiki generate --use-joern --exclude="*test*,tests/*"
```

### 2. Performance Monitoring

Track analysis performance:

```bash
codewiki generate --use-joern --verbose 2>&1 | grep -E "(analysis_time|joern_enhanced)"
```

### 3. Integration with CI/CD

```yaml
# GitHub Actions example
- name: Generate Documentation with Joern
  run: |
    codewiki generate --use-joern --github-pages
```

## Limitations

### Current Limitations
- **Language Support**: Best support for Python, Java, JavaScript/TypeScript
- **Performance**: 2-3x slower than AST-only analysis
- **Memory**: Higher memory usage for large codebases
- **Java Required**: Must have Java 19+ installed

### Planned Improvements
- **Caching**: Reuse Joern CPG between runs
- **Parallel Processing**: Multi-file analysis
- **Selective Joern**: Apply only to specific file types
- **Streaming**: Real-time analysis for large repositories

## Contributing

### Testing Joern Integration

```bash
# Test with sample repository
python -c "
from codewiki.src.be.dependency_analyzer.simplified_joern import run_performance_baseline
import json
result = run_performance_baseline('.')
print(json.dumps(result, indent=2))
"

# Test hybrid service
python -c "
from codewiki.src.be.dependency_analyzer.hybrid_analysis_service import HybridAnalysisService
service = HybridAnalysisService(enable_joern=True)
result = service.analyze_repository_hybrid('.', max_files=10)
print(f'Analysis complete: {result.get(\"metadata\", {}).get(\"joern_enhanced\", False)}')
"
```

## Support

For issues with Joern integration:

1. **Check system requirements**: Java 19+, sufficient memory
2. **Use verbose mode**: `--verbose` for detailed logs
3. **Check GitHub issues**: [CodeWiki Issues](https://github.com/yourusername/codewiki/issues)
4. **Report issues**: Include system info, repository size, and verbose logs

## Best Practices

### 1. Repository Structure
```
my-project/
├── joern.jar          # Auto-downloaded or manual
├── src/               # Source code
├── docs/              # Generated documentation
└── .gitignore         # Ignore temporary files
```

### 2. Development Workflow
```bash
# 1. Initial documentation (fast)
codewiki generate

# 2. Enhanced insights (slower, more detailed)
codewiki generate --use-joern

# 3. Production deployment
codewiki generate --create-branch --github-pages
```

### 3. Performance Optimization
- Use AST-only mode for frequent updates
- Use Joern mode for comprehensive analysis
- Consider repository size and analysis frequency

---

**Enjoy enhanced code insights with Joern integration! 🚀**