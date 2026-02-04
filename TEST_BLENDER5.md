# Blender 5 헤드리스 렌더링 테스트 가이드

## 수정 사항 요약

Blender 5 헤드리스 모드에서 렌더링이 멈추는 문제를 해결하기 위해 다음 수정을 적용했습니다:

### 1. 렌더링 호출 방식 변경
- **변경 전**: `bpy.ops.render.render(write_still=True)`
- **변경 후**: `bpy.ops.render.render("EXEC_DEFAULT", write_still=True)`
- **효과**: Blender 5 헤드리스 모드에서 멈춤 방지

### 2. Blender 실행 명령어 개선
- `--factory-startup` 플래그 추가: 애드온 충돌 방지
- 환경 변수 개선: Windows/Linux 환경에 맞게 설정

### 3. View Layer 업데이트 추가
- 렌더링 전 `bpy.context.view_layer.update()` 호출 추가

## 테스트 방법

### 방법 1: 배치 파일 사용 (Windows)

```batch
test_blender5_headless.bat
```

배치 파일에서 Blender 경로를 실제 설치 경로로 수정하세요.

### 방법 2: 직접 명령어 실행

```bash
# Windows
blender.exe --background --factory-startup empty_scene.blend --python data/static_scene/generator_script.py -- test_simple_blender.py output_dir

# Linux/Mac
blender --background --factory-startup empty_scene.blend --python data/static_scene/generator_script.py -- test_simple_blender.py output_dir
```

### 방법 3: Python 스크립트로 테스트

```python
python test_blender_render.py
```

## 수정된 파일 목록

### 렌더 스크립트 (11개)
- `data/static_scene/generator_script.py`
- `data/static_scene/generator_init_script.py`
- `data/dynamic_scene/generator_script.py`
- `data/blendergym/generator_script.py`
- `data/blendergym/single_render_script.py`
- `data/blendergym/pipeline_render_script.py`
- `data/blendergym/python_script.py`
- `data/blendergym/eval_render_script.py`
- `data/blendergym/all_render_script.py`
- `data/blenderbench/generator_script.py`

### Blender 실행 로직 (4개)
- `tools/blender/exec.py`
- `tools/blender/investigator_core.py`
- `runners/shared/blender_executor.py`
- `runners/blendergym/baseline.py`

## 확인 사항

렌더링이 성공적으로 완료되면:
1. 출력 디렉토리에 PNG 이미지 파일이 생성되어야 합니다
2. Blender 프로세스가 정상적으로 종료되어야 합니다 (멈추지 않아야 함)
3. 에러 메시지가 없어야 합니다

## 문제 해결

만약 여전히 렌더링이 멈춘다면:
1. Blender 버전 확인: `blender --version` (5.0 이상인지 확인)
2. GPU 드라이버 확인: CUDA/OPTIX 지원 여부 확인
3. 로그 확인: stdout/stderr 출력 확인
4. 타임아웃 설정: `tools/blender/exec.py`의 timeout 값 확인 (현재 600초)
